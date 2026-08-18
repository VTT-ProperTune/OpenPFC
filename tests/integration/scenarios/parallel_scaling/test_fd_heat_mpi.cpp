// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <mpi.h>
#include <numbers>
#include <vector>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/comm_sparse_exchange.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
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

  constexpr int halo_width = 1;
  auto u = data::field_from_subdomain<double>(decomp, rank, halo_width);
  u.for_each_owned([&](int i, int j, int k) { u(i, j, k) = 1.0; });

  comm::HaloExchange<HostSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD);
  halo.exchange();

  bool all_values_are_zero = true;
  u.for_each_owned([&](int i, int j, int k) {
    const double lap = u(i + 1, j, k) + u(i - 1, j, k) + u(i, j + 1, k) +
                       u(i, j - 1, k) + u(i, j, k + 1) + u(i, j, k - 1) -
                       6.0 * u(i, j, k);
    all_values_are_zero &= std::abs(lap) <= 1e-12;
  });
  REQUIRE(all_values_are_zero);
}

TEST_CASE("HaloExchange start/finish matches blocking face sync",
          "[MPI][fd]") {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Two ranks split along Z so ±Z neighbors are always different ranks.
  if (size != 2) {
    return;
  }

  auto world = world::uniform(16, 1.0);
  auto decomp = decomposition::create(world, {1, 1, 2});

  constexpr int halo_width = 1;
  auto a = data::field_from_subdomain<double>(decomp, rank, halo_width);
  auto b = data::field_from_subdomain<double>(decomp, rank, halo_width);
  const auto fill = static_cast<double>(rank);
  a.for_each_owned([&](int i, int j, int k) { a(i, j, k) = fill; });
  b.for_each_owned([&](int i, int j, int k) { b(i, j, k) = fill; });

  comm::HaloExchange<HostSpace, double> hex(a, decomp, rank, MPI_COMM_WORLD);
  comm::HaloExchange<HostSpace, double> pex(b, decomp, rank, MPI_COMM_WORLD);
  hex.exchange();
  pex.start();
  pex.finish();

  bool fields_match = true;
  const auto n = a.local_size();
  for (int k = -halo_width; k < n[2] + halo_width; ++k)
    for (int j = -halo_width; j < n[1] + halo_width; ++j)
      for (int i = -halo_width; i < n[0] + halo_width; ++i)
        fields_match &= std::abs(b(i, j, k) - a(i, j, k)) <= 1e-12;
  REQUIRE(fields_match);
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
  const double dx = 2.0 * std::numbers::pi / static_cast<double>(N);
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
  bool laplacian_matches = true;
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
  comm::SparseExchange<HostSpace, double> sex(
      u.data(), u.size(), decomp, rank, MPI_COMM_WORLD, halo_width);
  sex.exchange();
  halo::copy_to_face_layout(sex.halos(), face_halos);

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
        laplacian_matches &= std::abs(lap[c] - expected) <= 0.05;
      }
    }
  }
  REQUIRE(laplacian_matches);
}
