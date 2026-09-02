// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cstdint>
#include <cstring>

#include <catch2/catch_all.hpp>

#include <kobayashi/cli.hpp>
#include <kobayashi/openmp_engine.hpp>
#include <kobayashi/verification_utilities.hpp>

namespace {
[[nodiscard]] int ulp_distance(double a, double b) {
  std::uint64_t ua = 0;
  std::uint64_t ub = 0;
  std::memcpy(&ua, &a, sizeof(ua));
  std::memcpy(&ub, &b, sizeof(ub));
  const auto ia = static_cast<std::int64_t>(ua);
  const auto ib = static_cast<std::int64_t>(ub);
  const auto d = ia > ib ? ia - ib : ib - ia;
  return d > 0x7fffffff ? 0x7fffffff : static_cast<int>(d);
}
} // namespace

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
  // phi checksums are bitwise across Tohtori gcc / LUMI Cray. T sums move by
  // up to 2 ULP on Cray GNU (job 21683102); see BASELINES.md.
  REQUIRE(sp.sum == 0x1.b96bf451009d9p+3);
  REQUIRE(sp.sumsq == 0x1.4e770b1504ae4p+3);
  REQUIRE(ulp_distance(sT.sum, 0x1.6e128af4d5ac6p+0) <= 2);
  REQUIRE(ulp_distance(sT.sumsq, 0x1.6546ee0a021fp-2) <= 2);
}
