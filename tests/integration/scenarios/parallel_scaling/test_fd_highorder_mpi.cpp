// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <vector>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/decomposition/separated_halo_exchange.hpp>
#include <openpfc/kernel/field/finite_difference.hpp>

using namespace pfc;

TEST_CASE("4th/6th order separated Laplacian of constant field is zero",
          "[MPI][fd]") {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    return;
  }

  auto world = world::uniform(32, 1.0);
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

  constexpr int hw4 = 2;
  auto face4 = halo::allocate_face_halos<double>(decomp, rank, hw4);
  SeparatedFaceHaloExchanger<double> ex4(decomp, rank, hw4, MPI_COMM_WORLD);
  ex4.exchange_halos(u.data(), u.size(), face4);
  std::vector<double> lap4(nlocal, 0.0);
  std::array<const double *, 6> fp4{};
  for (int i = 0; i < 6; ++i) {
    fp4[static_cast<size_t>(i)] = face4[static_cast<size_t>(i)].data();
  }
  field::fd::laplacian_4th_order_interior_separated(u.data(), fp4, lap4.data(), nx,
                                                    ny, nz, inv, inv, inv, hw4);

  constexpr int hw6 = 3;
  auto face6 = halo::allocate_face_halos<double>(decomp, rank, hw6);
  SeparatedFaceHaloExchanger<double> ex6(decomp, rank, hw6, MPI_COMM_WORLD);
  ex6.exchange_halos(u.data(), u.size(), face6);
  std::vector<double> lap6(nlocal, 0.0);
  std::array<const double *, 6> fp6{};
  for (int i = 0; i < 6; ++i) {
    fp6[static_cast<size_t>(i)] = face6[static_cast<size_t>(i)].data();
  }
  field::fd::laplacian_6th_order_interior_separated(u.data(), fp6, lap6.data(), nx,
                                                    ny, nz, inv, inv, inv, hw6);

  const int imin4 = hw4;
  const int imax4 = nx - hw4;
  const int jmin4 = hw4;
  const int jmax4 = ny - hw4;
  const int kmin4 = hw4;
  const int kmax4 = nz - hw4;
  const int sxy = nx * ny;
  for (int iz = kmin4; iz < kmax4; ++iz) {
    for (int iy = jmin4; iy < jmax4; ++iy) {
      for (int ix = imin4; ix < imax4; ++ix) {
        const size_t c = static_cast<size_t>(ix) +
                         static_cast<size_t>(iy) * static_cast<size_t>(nx) +
                         static_cast<size_t>(iz) * static_cast<size_t>(sxy);
        REQUIRE(lap4[c] == Catch::Approx(0.0).margin(1e-11));
      }
    }
  }

  const int imin6 = hw6;
  const int imax6 = nx - hw6;
  const int jmin6 = hw6;
  const int jmax6 = ny - hw6;
  const int kmin6 = hw6;
  const int kmax6 = nz - hw6;
  for (int iz = kmin6; iz < kmax6; ++iz) {
    for (int iy = jmin6; iy < jmax6; ++iy) {
      for (int ix = imin6; ix < imax6; ++ix) {
        const size_t c = static_cast<size_t>(ix) +
                         static_cast<size_t>(iy) * static_cast<size_t>(nx) +
                         static_cast<size_t>(iz) * static_cast<size_t>(sxy);
        REQUIRE(lap6[c] == Catch::Approx(0.0).margin(1e-11));
      }
    }
  }
}

TEST_CASE("4th order Laplacian of global quadratic field is exact (discrete = 6)",
          "[MPI][fd]") {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    return;
  }

  auto world = world::uniform(32, 1.0);
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
  constexpr int hw2 = 2;
  auto face2 = halo::allocate_face_halos<double>(decomp, rank, hw2);
  SeparatedFaceHaloExchanger<double> ex2(decomp, rank, hw2, MPI_COMM_WORLD);
  ex2.exchange_halos(u.data(), u.size(), face2);
  std::vector<double> lap4(nlocal, 0.0);
  std::array<const double *, 6> fp2{};
  for (int i = 0; i < 6; ++i) {
    fp2[static_cast<size_t>(i)] = face2[static_cast<size_t>(i)].data();
  }
  field::fd::laplacian_4th_order_interior_separated(u.data(), fp2, lap4.data(), nx,
                                                    ny, nz, inv, inv, inv, hw2);

  const int imin = hw2;
  const int imax = nx - hw2;
  const int jmin = hw2;
  const int jmax = ny - hw2;
  const int kmin = hw2;
  const int kmax = nz - hw2;
  for (int iz = kmin; iz < kmax; ++iz) {
    for (int iy = jmin; iy < jmax; ++iy) {
      for (int ix = imin; ix < imax; ++ix) {
        const size_t c = static_cast<size_t>(ix) +
                         static_cast<size_t>(iy) * static_cast<size_t>(nx) +
                         static_cast<size_t>(iz) * static_cast<size_t>(sxy);
        REQUIRE(lap4[c] == Catch::Approx(6.0).margin(1e-9));
      }
    }
  }
}
