// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>
#include <iostream>
#include <mpi.h>
#include <vector>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/decomposition/separated_halo_exchange.hpp>
#include <openpfc/kernel/field/finite_difference.hpp>

using namespace pfc;

/** \example 15_finite_difference_heat.cpp
 *
 * Multi-rank explicit heat equation \( \partial u / \partial t = D \nabla^2 u \)
 * on a fully periodic 3D box with a 7-point Laplacian. Uses **separated face
 * halos** (`SeparatedFaceHaloExchanger`): the **core** field is contiguous
 * `nx×ny×nz` subdomain data safe to pass to `fft.forward` / `fft.backward` on
 * the same decomposition without mixing ghost semantics. Ghost values live in
 * six side buffers filled each step before the stencil.
 *
 * The Laplacian is the templated brick `laplacian_periodic_separated<2>`,
 * which iterates the **full owned region** `[0, n)` along every axis and
 * looks up missing neighbors from the matching face-halo slab. That is the
 * right primitive for a periodic problem on a separated layout: it updates
 * every owned cell (including those at the owned-region edge) without ever
 * reading uninitialized ghost padding.
 *
 * Run: `mpirun -np 4 ./15_finite_difference_heat`
 */
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  constexpr int N = 32;
  constexpr double dx = 1.0;
  constexpr double D = 1.0;
  constexpr int n_steps = 40;
  const double inv_dx2 = 1.0 / (dx * dx);
  // Explicit stability (conservative): dt <= dx^2 / (6 D) in 3D for unit
  // second-order Laplacian.
  const double dt = 0.15 * dx * dx / (6.0 * D);

  auto world = world::uniform(N, dx);
  auto decomp = decomposition::create(world, nproc);

  const auto &local_world = decomposition::get_subworld(decomp, rank);
  auto local_size = world::get_size(local_world);
  auto local_lower = world::get_lower(local_world);
  const int nx = local_size[0];
  const int ny = local_size[1];
  const int nz = local_size[2];
  const size_t nlocal =
      static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);

  std::vector<double> u(nlocal);
  std::vector<double> lap(nlocal);

  constexpr int halo_width = 1;
  auto face_halos = halo::allocate_face_halos<double>(decomp, rank, halo_width);

  const double cx = 0.5 * static_cast<double>(N - 1);
  const double sigma = static_cast<double>(N) / 6.0;

  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const int gx = local_lower[0] + ix;
        const int gy = local_lower[1] + iy;
        const int gz = local_lower[2] + iz;
        const double rx = static_cast<double>(gx) - cx;
        const double ry = static_cast<double>(gy) - cx;
        const double rz = static_cast<double>(gz) - cx;
        const double r2 = rx * rx + ry * ry + rz * rz;
        const size_t idx = static_cast<size_t>(ix) +
                           static_cast<size_t>(iy) * static_cast<size_t>(nx) +
                           static_cast<size_t>(iz) * static_cast<size_t>(nx * ny);
        u[idx] = std::exp(-r2 / (2.0 * sigma * sigma));
      }
    }
  }

  SeparatedFaceHaloExchanger<double> exchanger(decomp, rank, halo_width,
                                               MPI_COMM_WORLD);

  std::array<const double *, 6> face_ptrs;
  for (int i = 0; i < 6; ++i) {
    face_ptrs[static_cast<size_t>(i)] = face_halos[static_cast<size_t>(i)].data();
  }

  for (int step = 0; step < n_steps; ++step) {
    exchanger.exchange_halos(u.data(), u.size(), face_halos);
    std::fill(lap.begin(), lap.end(), 0.0);
    field::fd::laplacian_periodic_separated<2>(u.data(), face_ptrs, lap.data(), nx,
                                               ny, nz, inv_dx2, inv_dx2, inv_dx2,
                                               halo_width);
    for (size_t i = 0; i < nlocal; ++i) {
      u[i] += dt * D * lap[i];
    }
  }

  double sum_u = 0.0;
  for (double v : u) {
    sum_u += v;
  }
  double sum_global = 0.0;
  MPI_Reduce(&sum_u, &sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "Finite-difference heat (MPI ranks=" << nproc << ", N=" << N
              << ", steps=" << n_steps << ", separated halos)\n";
    std::cout << "Global sum(u) after stepping: " << sum_global << "\n";
  }

  MPI_Finalize();
  return 0;
}
