// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <vector>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_factory.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/padded_halo_exchange.hpp>
#include <openpfc/kernel/field/brick_iteration.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>

#include <wave2d/cli.hpp>
#include <wave2d/wave_boundary.hpp>
#include <wave2d/wave_model.hpp>
#include <wave2d/wave_step_separated.hpp>

using Catch::Matchers::WithinAbs;

TEST_CASE("wave2d::kC is 1.0", "[wave2d]") {
  REQUIRE_THAT(wave2d::kC, WithinAbs(1.0, 1e-15));
}

TEST_CASE("WaveModel::rhs basic", "[wave2d][WaveModel]") {
  wave2d::WaveModel m;
  m.inv_dx2 = 1.0;
  m.inv_dy2 = 1.0;
  wave2d::WaveLaplacian lap{.lxx = 0.0, .lyy = 0.0};
  auto inc = m.rhs(0.0, 2.5, lap);
  REQUIRE_THAT(inc.du, WithinAbs(2.5, 1e-15));
  REQUIRE_THAT(inc.dv, WithinAbs(0.0, 1e-15));
  lap.lxx = 1.0;
  lap.lyy = -0.5;
  inc = m.rhs(0.0, 0.0, lap);
  REQUIRE_THAT(inc.dv, WithinAbs(0.5, 1e-15));
}

TEST_CASE("parse_y_bc", "[wave2d][cli]") {
  REQUIRE(wave2d::parse_y_bc("dirichlet"));
  REQUIRE(*wave2d::parse_y_bc("d") == wave2d::YBoundaryKind::Dirichlet);
  REQUIRE(*wave2d::parse_y_bc("neumann") == wave2d::YBoundaryKind::Neumann);
  REQUIRE_FALSE(wave2d::parse_y_bc("bogus").has_value());
}

TEST_CASE("parse_manual round-trip", "[wave2d][cli]") {
  const char *argv[] = {"wave2d_fd_manual", "32", "40", "10", "0.02",
                        "dirichlet",        "0"};
  const auto c = wave2d::parse_manual(7, const_cast<char **>(argv));
  REQUIRE(c);
  REQUIRE(c->Nx == 32);
  REQUIRE(c->Ny == 40);
  REQUIRE(c->y_bc == wave2d::YBoundaryKind::Dirichlet);
}

TEST_CASE("fill_y_physical_ghosts_padded Dirichlet mirrors", "[wave2d][bc]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  REQUIRE(rank == 0);

  constexpr int Nx = 8;
  constexpr int Ny = 8;
  auto world = pfc::world::create(pfc::GridSize({Nx, Ny, 1}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(world, 1);
  constexpr int hw = 1;
  pfc::field::PaddedBrick<double> u(decomp, rank, hw);
  u.apply([&](double, double, double) { return 0.0; });
  u(0, 0, 0) = 1.25;
  pfc::PaddedHaloExchanger<double> halo(decomp, rank, hw, MPI_COMM_WORLD);
  halo.exchange_halos(u.data(), u.size());
  wave2d::fill_y_physical_ghosts_padded(u, wave2d::YBoundaryKind::Dirichlet, Ny,
                                        0.0);
  REQUIRE_THAT(u(0, -1, 0), WithinAbs(-1.25, 1e-12));
}

TEST_CASE("step_wave_separated_order2_cpu short vs padded manual single rank",
          "[wave2d][integration]") {
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REQUIRE(nproc == 1);

  constexpr int Nx = 16;
  constexpr int Ny = 16;
  const double dt = 0.01;
  const int n_steps = 3;

  auto world = pfc::world::create(pfc::GridSize({Nx, Ny, 1}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(world, 1);

  constexpr int hw = 1;
  pfc::field::PaddedBrick<double> u_pad(decomp, rank, hw);
  pfc::field::PaddedBrick<double> v_pad(decomp, rank, hw);
  pfc::field::PaddedBrick<double> lap_pad(decomp, rank, hw);
  pfc::PaddedHaloExchanger<double> halo(decomp, rank, hw, MPI_COMM_WORLD);

  const double xc = 0.5 * static_cast<double>(Nx - 1);
  const double yc = 0.5 * static_cast<double>(Ny - 1);
  const double sigma = 2.0;
  u_pad.apply([&](double x, double y, double) {
    const double dx = x - xc;
    const double dy = y - yc;
    return std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
  });
  v_pad.apply([](double, double, double) { return 0.0; });
  halo.exchange_halos(u_pad.data(), u_pad.size());
  wave2d::fill_y_physical_ghosts_padded(u_pad, wave2d::YBoundaryKind::Neumann, Ny,
                                        0.0);
  wave2d::WaveModel model;
  model.inv_dx2 = 1.0;
  model.inv_dy2 = 1.0;

  for (int s = 0; s < n_steps; ++s) {
    (void)s;
    halo.exchange_halos(u_pad.data(), u_pad.size());
    wave2d::fill_y_physical_ghosts_padded(u_pad, wave2d::YBoundaryKind::Neumann, Ny,
                                          0.0);
    pfc::field::for_each_owned(u_pad, [&](int i, int j, int k) {
      const double lxx =
          u_pad(i + 1, j, k) - 2.0 * u_pad(i, j, k) + u_pad(i - 1, j, k);
      const double lyy =
          u_pad(i, j + 1, k) - 2.0 * u_pad(i, j, k) + u_pad(i, j - 1, k);
      lap_pad(i, j, k) = model.inv_dx2 * lxx + model.inv_dy2 * lyy;
    });
    pfc::field::for_each_owned(u_pad, [&](int i, int j, int k) {
      const double v0 = v_pad(i, j, k);
      const double l = lap_pad(i, j, k);
      u_pad(i, j, k) += dt * v0;
      v_pad(i, j, k) += dt * wave2d::kC * wave2d::kC * l;
    });
  }

  const auto &local = pfc::decomposition::get_subworld(decomp, rank);
  const auto lo = pfc::world::get_lower(local);
  const auto sz = pfc::world::get_size(local);
  const int nx = sz[0];
  const int ny = sz[1];
  const int nz = sz[2];
  const std::size_t nlocal = static_cast<std::size_t>(nx) *
                             static_cast<std::size_t>(ny) *
                             static_cast<std::size_t>(nz);
  std::vector<double> u_sep(nlocal);
  std::vector<double> v_sep(nlocal, 0.0);
  std::vector<double> lap_sep(nlocal);
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const std::size_t idx =
            static_cast<std::size_t>(i) +
            static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(k) * static_cast<std::size_t>(nx * ny);
        const double x = static_cast<double>(lo[0] + i);
        const double y = static_cast<double>(lo[1] + j);
        const double dx = x - xc;
        const double dy = y - yc;
        u_sep[idx] = std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
      }
    }
  }
  auto face_halos = pfc::halo::allocate_face_halos<double>(decomp, rank, 1);
  pfc::SeparatedFaceHaloExchanger<double> exch(decomp, rank, 1, MPI_COMM_WORLD);
  for (int s = 0; s < n_steps; ++s) {
    wave2d::step_wave_separated_order2_cpu(u_sep, v_sep, lap_sep, face_halos, exch,
                                           nx, ny, nz, decomp, rank, dt,
                                           wave2d::YBoundaryKind::Neumann, Ny, 0.0);
  }

  double max_diff = 0.0;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const std::size_t idx =
            static_cast<std::size_t>(i) +
            static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(k) * static_cast<std::size_t>(nx * ny);
        max_diff = std::max(max_diff, std::abs(u_sep[idx] - u_pad(i, j, k)));
        max_diff = std::max(max_diff, std::abs(v_sep[idx] - v_pad(i, j, k)));
      }
    }
  }
  REQUIRE(max_diff < 1e-9);
}

int main(int argc, char *argv[]) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cerr << "test_wave2d: MPI_Init failed\n";
    return 1;
  }
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
