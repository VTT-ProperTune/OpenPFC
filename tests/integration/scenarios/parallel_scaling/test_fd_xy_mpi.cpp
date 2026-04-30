// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <mpi.h>
#include <numbers>
#include <vector>

#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_factory.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_exchange.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/decomposition/separated_halo_exchange.hpp>
#include <openpfc/kernel/field/finite_difference.hpp>

using namespace pfc;

namespace {

double periodic_xy_value(int gx, int gy, int nx_glob, int ny_glob) {
  gx = (gx + nx_glob) % nx_glob;
  gy = (gy + ny_glob) % ny_glob;
  return 10.0 * static_cast<double>(gy) + static_cast<double>(gx);
}

double expected_periodic_xy_laplacian(int gx, int gy, int nx_glob, int ny_glob) {
  const double uc = periodic_xy_value(gx, gy, nx_glob, ny_glob);
  return periodic_xy_value(gx + 1, gy, nx_glob, ny_glob) +
         periodic_xy_value(gx - 1, gy, nx_glob, ny_glob) +
         periodic_xy_value(gx, gy + 1, nx_glob, ny_glob) +
         periodic_xy_value(gx, gy - 1, nx_glob, ny_glob) - 4.0 * uc;
}

} // namespace

TEST_CASE("5-point XY Laplacian of constant field is zero (nz=1)", "[MPI][fd][xy]") {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Decomposition is hard-wired to {2,1,1}; only run when launched on
  // exactly 2 ranks (CI: `mpi_2procs_all`). Other rank counts skip.
  if (size != 2) {
    return;
  }

  auto world = world::create(GridSize({24, 24, 1}), PhysicalOrigin({0.0, 0.0, 0.0}),
                             GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(world, {2, 1, 1});

  const auto &local_world = decomposition::get_subworld(decomp, rank);
  auto local_size = world::get_size(local_world);
  const int nx = local_size[0];
  const int ny = local_size[1];
  const int nz = local_size[2];
  REQUIRE(nz == 1);

  const size_t nlocal =
      static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);

  std::vector<double> u(nlocal, 1.0);
  std::vector<double> lap(nlocal, 0.0);

  constexpr int halo_width = 1;
  HaloExchanger<double> exchanger(decomp, rank, halo_width, MPI_COMM_WORLD);
  exchanger.exchange_halos(u.data(), u.size());

  const double inv = 1.0;
  field::fd::laplacian2d_xy_interior<2>(u.data(), lap.data(), nx, ny, nz, inv, inv,
                                        halo_width);

  for (size_t i = 0; i < nlocal; ++i) {
    REQUIRE(lap[i] == Catch::Approx(0.0).margin(1e-12));
  }
}

TEST_CASE("laplacian2d_xy_periodic_separated<2> matches analytic 2D Laplacian on "
          "every owned cell (MPI)",
          "[MPI][fd][xy]") {
  // 2D analogue of the 3D analytic test in `test_fd_heat_mpi.cpp`. Sample
  // u(x,y) = sin(x) cos(y) on a fully periodic [0, 2π)^2 box decomposed
  // as {2, 1, 1} (nz = 1); analytic Laplacian is Δu = -2 u. Each rank
  // checks the templated brick against the analytic value at every owned
  // cell, including the owned-region edges.
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

  auto world = world::create(GridSize({N, N, 1}), PhysicalOrigin({0.0, 0.0, 0.0}),
                             GridSpacing({dx, dx, dx}));
  auto decomp = decomposition::create(world, {2, 1, 1});

  const auto &local_world = decomposition::get_subworld(decomp, rank);
  auto local_size = world::get_size(local_world);
  auto local_lower = world::get_lower(local_world);
  const int nx = local_size[0];
  const int ny = local_size[1];
  const int nz = local_size[2];
  REQUIRE(nz == 1);

  const size_t nlocal =
      static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
  std::vector<double> u(nlocal);
  std::vector<double> lap(nlocal, 0.0);
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      const double x = static_cast<double>(local_lower[0] + ix) * dx;
      const double y = static_cast<double>(local_lower[1] + iy) * dx;
      const size_t idx = static_cast<size_t>(ix) +
                         static_cast<size_t>(iy) * static_cast<size_t>(nx);
      u[idx] = std::sin(x) * std::cos(y);
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
  field::fd::laplacian2d_xy_periodic_separated<2>(
      u.data(), face_ptrs, lap.data(), nx, ny, nz, inv_dx2, inv_dx2, halo_width);

  // Second-order central FD on a smooth periodic test gives an O(dx^2)
  // truncation error; with N = 32 and |Δu| ≤ 2 the absolute tolerance
  // 0.05 is comfortable.
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      const size_t c = static_cast<size_t>(ix) +
                       static_cast<size_t>(iy) * static_cast<size_t>(nx);
      const double expected = -2.0 * u[c];
      REQUIRE(lap[c] == Catch::Approx(expected).margin(0.05));
    }
  }
}

TEST_CASE("Separated face halos contain opposite periodic neighbor faces (XY)",
          "[MPI][fd][xy][ring]") {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 3) {
    return;
  }

  constexpr int nx_glob = 12;
  constexpr int ny_glob = 5;
  auto world =
      world::create(GridSize({nx_glob, ny_glob, 1}), PhysicalOrigin({0.0, 0.0, 0.0}),
                    GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(world, {3, 1, 1});

  const auto &local_world = decomposition::get_subworld(decomp, rank);
  auto local_size = world::get_size(local_world);
  auto local_lower = world::get_lower(local_world);
  const int nx = local_size[0];
  const int ny = local_size[1];
  const int nz = local_size[2];
  REQUIRE(nz == 1);

  std::vector<double> u(static_cast<size_t>(nx) * static_cast<size_t>(ny));
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      const int gx = local_lower[0] + ix;
      const int gy = local_lower[1] + iy;
      u[static_cast<size_t>(ix) +
        static_cast<size_t>(iy) * static_cast<size_t>(nx)] =
          periodic_xy_value(gx, gy, nx_glob, ny_glob);
    }
  }

  constexpr int halo_width = 1;
  auto face_halos = halo::allocate_face_halos<double>(decomp, rank, halo_width);
  SeparatedFaceHaloExchanger<double> exchanger(decomp, rank, halo_width,
                                               MPI_COMM_WORLD);
  exchanger.exchange_halos(u.data(), u.size(), face_halos);

  for (int iy = 0; iy < ny; ++iy) {
    const int gy = local_lower[1] + iy;
    const auto offset = static_cast<size_t>(iy);
    REQUIRE(
        face_halos[0][offset] ==
        Catch::Approx(periodic_xy_value(local_lower[0] + nx, gy, nx_glob, ny_glob))
            .margin(1e-12));
    REQUIRE(
        face_halos[1][offset] ==
        Catch::Approx(periodic_xy_value(local_lower[0] - 1, gy, nx_glob, ny_glob))
            .margin(1e-12));
  }
}

TEST_CASE("5-point XY separated periodic Laplacian matches serial global formula",
          "[MPI][fd][xy][multiple]") {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 2 && size != 4) {
    return;
  }

  constexpr int nx_glob = 12;
  constexpr int ny_glob = 12;
  auto world =
      world::create(GridSize({nx_glob, ny_glob, 1}), PhysicalOrigin({0.0, 0.0, 0.0}),
                    GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = (size == 2) ? decomposition::create(world, {2, 1, 1})
                            : decomposition::create(world, {2, 2, 1});

  const auto &local_world = decomposition::get_subworld(decomp, rank);
  auto local_size = world::get_size(local_world);
  auto local_lower = world::get_lower(local_world);
  const int nx = local_size[0];
  const int ny = local_size[1];
  const int nz = local_size[2];
  REQUIRE(nz == 1);

  const size_t nlocal =
      static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
  std::vector<double> u(nlocal);
  std::vector<double> lap(nlocal, 0.0);
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      const int gx = local_lower[0] + ix;
      const int gy = local_lower[1] + iy;
      u[static_cast<size_t>(ix) +
        static_cast<size_t>(iy) * static_cast<size_t>(nx)] =
          periodic_xy_value(gx, gy, nx_glob, ny_glob);
    }
  }

  constexpr int halo_width = 1;
  auto face_halos = halo::allocate_face_halos<double>(decomp, rank, halo_width);
  SeparatedFaceHaloExchanger<double> exchanger(decomp, rank, halo_width,
                                               MPI_COMM_WORLD);
  exchanger.exchange_halos(u.data(), u.size(), face_halos);

  std::array<const double *, 6> face_ptrs{};
  for (int i = 0; i < 6; ++i) {
    face_ptrs[static_cast<size_t>(i)] = face_halos[static_cast<size_t>(i)].data();
  }
  field::fd::laplacian2d_xy_periodic_separated<2>(u.data(), face_ptrs, lap.data(),
                                                  nx, ny, nz, 1.0, 1.0, halo_width);

  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      const int gx = local_lower[0] + ix;
      const int gy = local_lower[1] + iy;
      const size_t c = static_cast<size_t>(ix) +
                       static_cast<size_t>(iy) * static_cast<size_t>(nx);
      REQUIRE(lap[c] ==
              Catch::Approx(expected_periodic_xy_laplacian(gx, gy, nx_glob, ny_glob))
                  .margin(1e-12));
    }
  }
}
