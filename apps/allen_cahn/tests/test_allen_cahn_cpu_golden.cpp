// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <vector>

#include <allen_cahn/common.hpp>
#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>

using Catch::Matchers::WithinRel;

TEST_CASE("Allen–Cahn CPU golden matches CPU-vs-CUDA config",
          "[AllenCahn][cpu_golden][parity]") {
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REQUIRE(nproc == 1);

  allen_cahn::RunConfig cfg;
  cfg.nx_glob = 32;
  cfg.ny_glob = 32;
  cfg.n_steps = 20;
  cfg.dt = 0.002;
  cfg.M = 1.0;
  cfg.epsilon = 0.5;
  cfg.driving_force = 0.25;

  auto domain = pfc::domain::create(pfc::GridSize({cfg.nx_glob, cfg.ny_glob, 1}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  const auto &local_box = pfc::decomposition::local_box(decomp, rank);
  auto local_size = local_box.size;
  const int nx = local_size[0];
  const int ny = local_size[1];
  const int nz = local_size[2];
  REQUIRE(nz == 1);
  const std::size_t nlocal = static_cast<std::size_t>(nx) *
                             static_cast<std::size_t>(ny) *
                             static_cast<std::size_t>(nz);

  const double dx = 1.0;
  const double inv_dx2 = 1.0 / (dx * dx);
  const double inv_dy2 = inv_dx2;
  const double inv_eps2 = 1.0 / (cfg.epsilon * cfg.epsilon);

  std::vector<double> u(nlocal);
  std::vector<double> lap(nlocal);
  allen_cahn::fill_initial_condition(&u, decomp, rank);

  constexpr int halo_width = allen_cahn::RunConfig::kHaloWidth;
  auto face = pfc::halo::allocate_face_halos<double>(decomp, rank, halo_width);
  pfc::comm::SparseExchange<pfc::HostSpace, double> exch(
      u.data(), u.size(), decomp, rank, MPI_COMM_WORLD, halo_width);

  for (int step = 0; step < cfg.n_steps; ++step) {
    allen_cahn::step_explicit_euler_cpu(&u, &lap, &face, &exch, nx, ny, nz, inv_dx2,
                                        inv_dy2, cfg.dt, cfg.M, inv_eps2,
                                        cfg.driving_force);
  }

  double sum = 0.0;
  double sumsq = 0.0;
  for (double x : u) {
    sum += x;
    sumsq += x * x;
  }
  if (rank == 0) {
    std::cout << std::setprecision(17) << "CPU_GOLDEN allen_cahn n=" << nlocal
              << " sum=" << sum << " sumsq=" << sumsq << '\n';
  }
  REQUIRE(nlocal == 1024);
  REQUIRE(std::isfinite(sum));
  REQUIRE(std::isfinite(sumsq));
  // Tohtori g0005, gcc 15.2 Debug, same config as test_allen_cahn_cpu_vs_cuda.
  REQUIRE_THAT(sum, WithinRel(-967.34722270794146, 1e-10));
  REQUIRE_THAT(sumsq, WithinRel(961.14566882919667, 1e-10));
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
