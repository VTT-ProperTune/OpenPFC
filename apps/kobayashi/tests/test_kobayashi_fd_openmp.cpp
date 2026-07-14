// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_all.hpp>

#include <kobayashi/cli.hpp>
#include <kobayashi/openmp_engine.hpp>

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
