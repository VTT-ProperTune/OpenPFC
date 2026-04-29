// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <mpi.h>
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

  if (size < 2) {
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
  field::fd::laplacian_5point_xy_interior(u.data(), lap.data(), nx, ny, nz, inv, inv,
                                          halo_width);

  for (size_t i = 0; i < nlocal; ++i) {
    REQUIRE(lap[i] == Catch::Approx(0.0).margin(1e-12));
  }
}

TEST_CASE("5-point XY separated vs in-place Laplacian on interior (nz=1)",
          "[MPI][fd][xy]") {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
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

  std::vector<double> u_inplace(nlocal);
  std::vector<double> u_core(nlocal);
  for (size_t i = 0; i < nlocal; ++i) {
    const double v = static_cast<double>(rank) + 0.01 * static_cast<double>(i);
    u_inplace[i] = v;
    u_core[i] = v;
  }

  constexpr int halo_width = 1;
  std::vector<double> lap_in(nlocal, 0.0);
  std::vector<double> lap_sep(nlocal, 0.0);

  HaloExchanger<double> hex(decomp, rank, halo_width, MPI_COMM_WORLD);
  hex.exchange_halos_packed(u_inplace.data(), u_inplace.size());
  const double inv = 1.0;
  field::fd::laplacian_5point_xy_interior(u_inplace.data(), lap_in.data(), nx, ny,
                                          nz, inv, inv, halo_width);

  auto face_halos = halo::allocate_face_halos<double>(decomp, rank, halo_width);
  SeparatedFaceHaloExchanger<double> sex(decomp, rank, halo_width, MPI_COMM_WORLD);
  sex.exchange_halos(u_core.data(), u_core.size(), face_halos);

  std::array<const double *, 6> face_ptrs;
  for (int i = 0; i < 6; ++i) {
    face_ptrs[static_cast<size_t>(i)] = face_halos[static_cast<size_t>(i)].data();
  }
  field::fd::laplacian_5point_xy_interior_separated(
      u_core.data(), face_ptrs, lap_sep.data(), nx, ny, nz, inv, inv, halo_width);

  // Compare only cells whose stencil uses owned core values in both layouts.
  // Halo-touching boundary cells are covered by the explicit face-halo and
  // periodic-Laplacian tests below.
  const int imin = 2 * halo_width;
  const int imax = nx - 2 * halo_width;
  const int jmin = 2 * halo_width;
  const int jmax = ny - 2 * halo_width;
  constexpr int iz = 0;
  const int sxy = nx * ny;
  for (int iy = jmin; iy < jmax; ++iy) {
    for (int ix = imin; ix < imax; ++ix) {
      const size_t c = static_cast<size_t>(ix) +
                       static_cast<size_t>(iy) * static_cast<size_t>(nx) +
                       static_cast<size_t>(iz) * static_cast<size_t>(sxy);
      REQUIRE(lap_sep[c] == Catch::Approx(lap_in[c]).margin(1e-11));
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
  field::fd::laplacian_5point_xy_periodic_separated(
      u.data(), face_ptrs, lap.data(), nx, ny, nz, 1.0, 1.0, halo_width);

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
