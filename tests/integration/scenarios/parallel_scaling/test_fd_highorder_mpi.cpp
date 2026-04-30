// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <vector>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/field/finite_difference.hpp>

using namespace pfc;

TEST_CASE("even-order (2..20) interior Laplacian of constant field is zero",
          "[MPI][fd]") {
  // Sweep the runtime-order dispatcher `laplacian_interior(order, ...)`
  // across `order = 2..20` (step 2) on each MPI rank's owned brick. The
  // interior loop `[hw, n - hw)` reads only from `u`, so no halo
  // exchange is needed; the test focus is "every templated order maps a
  // constant field to zero on the interior, on every rank".
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Decomposition is hard-wired to {2,1,1}; only run when launched on
  // exactly 2 ranks (CI: `mpi_2procs_all`). Other rank counts skip.
  if (size != 2) {
    return;
  }

  auto world = world::uniform(128, 1.0);
  auto decomp = decomposition::create(world, {2, 1, 1});

  const auto &local_world = decomposition::get_subworld(decomp, rank);
  auto local_size = world::get_size(local_world);
  const int nx = local_size[0];
  const int ny = local_size[1];
  const int nz = local_size[2];
  const size_t nlocal =
      static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);

  std::vector<double> u(nlocal, 3.0);
  const double inv = 1.0;
  const int sxy = nx * ny;

  for (int order = 2; order <= 20; order += 2) {
    const int hw = order / 2;
    std::vector<double> lap(nlocal, 0.0);
    field::fd::laplacian_interior(order, u.data(), lap.data(), nx, ny, nz, inv, inv,
                                  inv, hw);

    const int imin = hw;
    const int imax = nx - hw;
    const int jmin = hw;
    const int jmax = ny - hw;
    const int kmin = hw;
    const int kmax = nz - hw;
    for (int iz = kmin; iz < kmax; ++iz) {
      for (int iy = jmin; iy < jmax; ++iy) {
        for (int ix = imin; ix < imax; ++ix) {
          const size_t c = static_cast<size_t>(ix) +
                           static_cast<size_t>(iy) * static_cast<size_t>(nx) +
                           static_cast<size_t>(iz) * static_cast<size_t>(sxy);
          INFO("order " << order << " rank " << rank);
          REQUIRE(lap[c] == Catch::Approx(0.0).margin(1e-10));
        }
      }
    }
  }
}

TEST_CASE("even-order (2..20) Laplacian of global quadratic is exact (6)",
          "[MPI][fd]") {
  // Sweep the runtime-order dispatcher on a globally-quadratic field
  // u(x,y,z) = gx^2 + gy^2 + gz^2. The exact analytic Laplacian is 6,
  // and the central even-order stencil is exact for polynomials of
  // degree <= order + 1, so every order in the table must reproduce 6
  // bit-for-bit on the interior slab on every rank.
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Decomposition is hard-wired to {2,1,1}; only run when launched on
  // exactly 2 ranks (CI: `mpi_2procs_all`). Other rank counts skip.
  if (size != 2) {
    return;
  }

  auto world = world::uniform(128, 1.0);
  auto decomp = decomposition::create(world, {2, 1, 1});

  const auto &local_world = decomposition::get_subworld(decomp, rank);
  auto local_size = world::get_size(local_world);
  auto local_lower = world::get_lower(local_world);
  const int nx = local_size[0];
  const int ny = local_size[1];
  const int nz = local_size[2];
  const size_t nlocal =
      static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);

  std::vector<double> u(nlocal);
  const int sxy = nx * ny;
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const int gx = local_lower[0] + ix;
        const int gy = local_lower[1] + iy;
        const int gz = local_lower[2] + iz;
        const size_t c = static_cast<size_t>(ix) +
                         static_cast<size_t>(iy) * static_cast<size_t>(nx) +
                         static_cast<size_t>(iz) * static_cast<size_t>(sxy);
        const double v = static_cast<double>(gx * gx + gy * gy + gz * gz);
        u[c] = v;
      }
    }
  }

  const double inv = 1.0;

  for (int order = 2; order <= 20; order += 2) {
    const int hw = order / 2;
    std::vector<double> lap(nlocal, 0.0);
    field::fd::laplacian_interior(order, u.data(), lap.data(), nx, ny, nz, inv, inv,
                                  inv, hw);

    const int imin = hw;
    const int imax = nx - hw;
    const int jmin = hw;
    const int jmax = ny - hw;
    const int kmin = hw;
    const int kmax = nz - hw;
    for (int iz = kmin; iz < kmax; ++iz) {
      for (int iy = jmin; iy < jmax; ++iy) {
        for (int ix = imin; ix < imax; ++ix) {
          const size_t c = static_cast<size_t>(ix) +
                           static_cast<size_t>(iy) * static_cast<size_t>(nx) +
                           static_cast<size_t>(iz) * static_cast<size_t>(sxy);
          INFO("order " << order << " rank " << rank);
          REQUIRE(lap[c] == Catch::Approx(6.0).margin(1e-8));
        }
      }
    }
  }
}
