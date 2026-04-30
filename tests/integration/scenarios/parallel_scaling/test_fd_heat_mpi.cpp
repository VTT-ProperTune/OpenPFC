// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <mpi.h>
#include <vector>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_exchange.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/decomposition/halo_persistent.hpp>
#include <openpfc/kernel/decomposition/separated_halo_exchange.hpp>
#include <openpfc/kernel/field/finite_difference.hpp>

using namespace pfc;

TEST_CASE("Laplacian of constant field is zero after halo exchange", "[MPI][fd]") {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Decomposition is hard-wired to {2,1,1}; only run when launched on
  // exactly 2 ranks (CI: `mpi_2procs_all`). Other rank counts skip.
  if (size != 2) {
    return;
  }

  auto world = world::uniform(24, 1.0);
  auto decomp = decomposition::create(world, {2, 1, 1});

  const auto &local_world = decomposition::get_subworld(decomp, rank);
  auto local_size = world::get_size(local_world);
  const int nx = local_size[0];
  const int ny = local_size[1];
  const int nz = local_size[2];
  const size_t nlocal =
      static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);

  std::vector<double> u(nlocal, 1.0);
  std::vector<double> lap(nlocal, 0.0);

  constexpr int halo_width = 1;
  HaloExchanger<double> exchanger(decomp, rank, halo_width, MPI_COMM_WORLD);
  exchanger.exchange_halos(u.data(), u.size());

  const double inv = 1.0;
  field::fd::laplacian_interior<2>(u.data(), lap.data(), nx, ny, nz, inv, inv, inv,
                                   halo_width);

  for (size_t i = 0; i < nlocal; ++i) {
    REQUIRE(lap[i] == Catch::Approx(0.0).margin(1e-12));
  }
}

TEST_CASE("PersistentHaloExchanger matches HaloExchanger face sync",
          "[MPI][fd][persistent]") {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 4) {
    return;
  }

  auto world = world::uniform(16, 1.0);
  auto decomp = decomposition::create(world, {2, 2, 1});

  const auto &local_world = decomposition::get_subworld(decomp, rank);
  auto local_size = world::get_size(local_world);
  const int nx = local_size[0];
  const int ny = local_size[1];
  const int nz = local_size[2];
  const size_t nlocal =
      static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);

  std::vector<double> a(nlocal);
  std::vector<double> b(nlocal);
  const auto fill = static_cast<double>(rank);
  for (size_t i = 0; i < nlocal; ++i) {
    a[i] = fill;
    b[i] = fill;
  }

  constexpr int halo_width = 1;
  HaloExchanger<double> hex(decomp, rank, halo_width, MPI_COMM_WORLD);
  PersistentHaloExchanger<double> pex(decomp, rank, halo_width, MPI_COMM_WORLD,
                                      b.data());

  hex.exchange_halos(a.data(), a.size());
  pex.exchange_halos();

  for (size_t i = 0; i < nlocal; ++i) {
    REQUIRE(b[i] == Catch::Approx(a[i]).margin(1e-12));
  }
}

TEST_CASE("laplacian_periodic_separated<2> matches analytic Laplacian on every "
          "owned cell (MPI)",
          "[MPI][fd][separated]") {
  // Sample u(x,y,z) = sin(x) cos(y) sin(z) on a fully periodic [0, 2π)^3
  // box decomposed as {2, 1, 1}; the analytic Laplacian is Δu = -3 u.
  // Each rank checks the templated brick against the analytic value at
  // every owned cell (including the owned-region edges, which is the
  // whole point of the periodic-separated form).
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Decomposition is hard-wired to {2,1,1}; only run when launched on
  // exactly 2 ranks (CI: `mpi_2procs_all`). Other rank counts skip.
  if (size != 2) {
    return;
  }

  constexpr int N = 32;
  const double dx = 2.0 * M_PI / static_cast<double>(N);
  const double inv_dx2 = 1.0 / (dx * dx);

  auto world = world::uniform(N, dx);
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
  std::vector<double> lap(nlocal, 0.0);
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const double x = static_cast<double>(local_lower[0] + ix) * dx;
        const double y = static_cast<double>(local_lower[1] + iy) * dx;
        const double z = static_cast<double>(local_lower[2] + iz) * dx;
        const size_t idx = static_cast<size_t>(ix) +
                           static_cast<size_t>(iy) * static_cast<size_t>(nx) +
                           static_cast<size_t>(iz) * static_cast<size_t>(nx) *
                               static_cast<size_t>(ny);
        u[idx] = std::sin(x) * std::cos(y) * std::sin(z);
      }
    }
  }

  constexpr int halo_width = 1;
  auto face_halos = halo::allocate_face_halos<double>(decomp, rank, halo_width);
  SeparatedFaceHaloExchanger<double> sex(decomp, rank, halo_width, MPI_COMM_WORLD);
  sex.exchange_halos(u.data(), u.size(), face_halos);

  std::array<const double *, 6> face_ptrs;
  for (int i = 0; i < 6; ++i) {
    face_ptrs[static_cast<size_t>(i)] = face_halos[static_cast<size_t>(i)].data();
  }
  field::fd::laplacian_periodic_separated<2>(u.data(), face_ptrs, lap.data(), nx, ny,
                                             nz, inv_dx2, inv_dx2, inv_dx2,
                                             halo_width);

  // Second-order central FD on a smooth periodic test gives an O(dx^2)
  // truncation error; with N = 32 and |Δu| ≤ 3 we comfortably stay
  // within an absolute tolerance of 0.05.
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const size_t c = static_cast<size_t>(ix) +
                         static_cast<size_t>(iy) * static_cast<size_t>(nx) +
                         static_cast<size_t>(iz) * static_cast<size_t>(nx) *
                             static_cast<size_t>(ny);
        const double expected = -3.0 * u[c];
        REQUIRE(lap[c] == Catch::Approx(expected).margin(0.05));
      }
    }
  }
}
