// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_all.hpp>

#include <kobayashi/cli.hpp>
#include <kobayashi/openmp_engine.hpp>
#include <kobayashi/verification_utilities.hpp>

TEST_CASE("kobayashi_fd_openmp thread parity vs serial") {
  kobayashi::RunConfigOpenMP base{};
  base.Nx = 56;
  base.Ny = 56;
  base.n_steps = 80;
  base.dt = 1.0e-4;
  base.dx = 0.03;
  base.output_dir = ".";
  base.num_threads = 1;

  auto ref = kobayashi::openmp_engine::run(base, /*skip_png=*/true, /*quiet=*/true);

  base.num_threads = 4;
  auto par = kobayashi::openmp_engine::run(base, /*skip_png=*/true, /*quiet=*/true);

  REQUIRE(ref.phi_xy == par.phi_xy);
  REQUIRE(ref.tempr_xy == par.tempr_xy);
}

TEST_CASE("kobayashi OpenMP 1-rank HEX matches MPI nproc=1 smoke") {
  kobayashi::RunConfigOpenMP cfg{};
  cfg.Nx = 32;
  cfg.Ny = 32;
  cfg.n_steps = 4;
  cfg.dt = 1.0e-4;
  cfg.dx = 0.03;
  cfg.output_dir = ".";
  cfg.num_threads = 1;

  auto res = kobayashi::openmp_engine::run(cfg, /*skip_png=*/true, /*quiet=*/true);
  const FieldStats sp = stats_global_ordered(res.phi_xy, cfg.Nx, cfg.Ny);
  const FieldStats sT = stats_global_ordered(res.tempr_xy, cfg.Nx, cfg.Ny);

  // Pinned from `kobayashi_fd_manual` nproc=1, same (Nx, Ny, steps, dt, dx).
  REQUIRE(sp.sum == 0x1.b96bf451009d9p+3);
  REQUIRE(sp.sumsq == 0x1.4e770b1504ae4p+3);
  REQUIRE(sT.sum == 0x1.6e128af4d5ac6p+0);
  REQUIRE(sT.sumsq == 0x1.6546ee0a021fp-2);
}
