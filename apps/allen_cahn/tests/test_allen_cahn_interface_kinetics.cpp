// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_allen_cahn_interface_kinetics.cpp
 * @brief The pass criterion must describe the physics, not the grid.
 *
 * @details
 * The app used to require the superlevel area to reach 5x its initial value.
 * The seed radius scales with the grid (`sigma = 0.055 * min(nx, ny)`) but the
 * interface speed does not, so `(R0 + v t)^2 / R0^2` shrinks as the grid grows
 * and the *same physics* scored 6.08x at 64^2, 3.50x at 128^2 and 2.55x at
 * 256^2 — pass, fail, fail. The grid-independent quantity is `dR/dt`, and the
 * headline test here is that two grid sizes agree on it and reach the same
 * verdict.
 *
 * The kinetics are read off the second half of the run: the Gaussian initial
 * condition is far from the equilibrium `tanh` and its collapse moves the
 * contour by a distance that scales with the seed, so a whole-run average
 * inherits the same grid dependence the area ratio had.
 *
 * Since 0.2 this file also pins the *preset*, not just the criterion. A
 * canonical demonstration whose interface is 0.76 cells wide and whose
 * driving force is 6% under the bistability ceiling is not a demonstration
 * of Allen–Cahn; it is a demonstration of two large errors cancelling. See
 * `docs/report/data/allen_cahn_resolution_margin.csv` for the measurements.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <mpi.h>
#include <vector>

#include <allen_cahn/common.hpp>
#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

constexpr double kDx = 1.0;

/// Run the shipped CPU update and return the four superlevel-area samples the
/// criterion reads. Single rank; mirrors `src/cpu/allen_cahn.cpp`.
allen_cahn::AreaSamples run_and_sample(const allen_cahn::RunConfig &cfg) {
  auto domain = pfc::domain::create(pfc::GridSize({cfg.nx_glob, cfg.ny_glob, 1}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({kDx, kDx, kDx}));
  auto decomp = pfc::decomposition::create(domain, 1);
  const auto &local_box = pfc::decomposition::local_box(decomp, 0);
  const auto local_size = local_box.size;
  const int nx = local_size[0];
  const int ny = local_size[1];
  const int nz = local_size[2];
  const auto nlocal = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                      static_cast<std::size_t>(nz);

  const double inv_dx2 = 1.0 / (kDx * kDx);
  const double inv_eps2 = 1.0 / (cfg.epsilon * cfg.epsilon);

  std::vector<double> u(nlocal);
  std::vector<double> lap(nlocal);
  allen_cahn::fill_initial_condition(&u, decomp, 0);

  constexpr int halo_width = allen_cahn::RunConfig::kHaloWidth;
  auto face = pfc::halo::allocate_face_halos<double>(decomp, 0, halo_width);
  pfc::comm::SparseExchange<pfc::HostSpace, double> exch(
      u.data(), u.size(), decomp, 0, MPI_COMM_WORLD, halo_width);

  allen_cahn::AreaSamples areas;
  areas.initial = allen_cahn::count_cells_above(
      u, allen_cahn::RunConfig::kLevelSetThreshold);
  const int step_half = cfg.n_steps / 2;
  const int step_three_quarter = (3 * cfg.n_steps) / 4;

  for (int step = 0; step < cfg.n_steps; ++step) {
    allen_cahn::step_explicit_euler_cpu(&u, &lap, &face, &exch, nx, ny, nz, inv_dx2,
                                        inv_dx2, cfg.dt, cfg.M, inv_eps2,
                                        cfg.driving_force);
    const int done = step + 1;
    if (done == step_half) {
      areas.half = allen_cahn::count_cells_above(
          u, allen_cahn::RunConfig::kLevelSetThreshold);
    }
    if (done == step_three_quarter) {
      areas.three_quarter = allen_cahn::count_cells_above(
          u, allen_cahn::RunConfig::kLevelSetThreshold);
    }
  }
  areas.final_ = allen_cahn::count_cells_above(
      u, allen_cahn::RunConfig::kLevelSetThreshold);
  return areas;
}

} // namespace

TEST_CASE("equivalent_radius inverts the area of a disc",
          "[AllenCahn][kinetics][unit]") {
  REQUIRE_THAT(allen_cahn::equivalent_radius(0, 1.0), WithinAbs(0.0, 1e-15));
  // A disc of radius 10 has area 100 pi; round-trip that back to 10.
  const auto cells =
      static_cast<std::int64_t>(std::lround(100.0 * 3.14159265358979323846));
  REQUIRE_THAT(allen_cahn::equivalent_radius(cells, 1.0), WithinRel(10.0, 1e-3));
  // dx scales lengths, so halving it halves the radius for the same cell count.
  REQUIRE_THAT(allen_cahn::equivalent_radius(cells, 0.5),
               WithinRel(0.5 * allen_cahn::equivalent_radius(cells, 1.0), 1e-12));
}

TEST_CASE("sharp-interface velocity follows (3/2) F eps sqrt(2M)",
          "[AllenCahn][kinetics][unit]") {
  // Shipped preset: M = 8, eps = 0.75, F = 0.25.
  REQUIRE_THAT(allen_cahn::sharp_interface_velocity(8.0, 0.75, 0.25),
               WithinRel(1.5 * 0.25 * 0.75 * 4.0, 1e-12));
  // Linear in the driving force, so no driving force means no front.
  REQUIRE_THAT(allen_cahn::sharp_interface_velocity(8.0, 0.75, 0.0),
               WithinAbs(0.0, 1e-15));
  // Interface half-width, in cells.
  REQUIRE_THAT(allen_cahn::interface_width_cells(8.0, 0.75, 1.0),
               WithinRel(3.0, 1e-9));
  // Curvature correction: a disc of radius R moves at v_flat - M/R, so a
  // disc of exactly the critical radius does not move at all.
  const double v_flat = allen_cahn::sharp_interface_velocity(8.0, 0.75, 0.25);
  const double r_star = allen_cahn::critical_radius_cells(8.0, 0.75, 0.25, 1.0);
  REQUIRE_THAT(r_star, WithinRel(8.0 / v_flat, 1e-12));
  REQUIRE_THAT(allen_cahn::curvature_corrected_velocity(v_flat, 8.0, r_star),
               WithinAbs(0.0, 1e-12));
  REQUIRE(allen_cahn::curvature_corrected_velocity(v_flat, 8.0, 0.5 * r_star) < 0.0);
  // A flat front (R -> infinity) recovers the uncorrected law.
  REQUIRE_THAT(allen_cahn::curvature_corrected_velocity(v_flat, 8.0, 1e12),
               WithinRel(v_flat, 1e-9));
}

/**
 * @brief The shipped preset has to be a preset a reader can trust.
 *
 * Both halves of this failed before the 0.2 preset move (`eps = 0.19`,
 * `F = 10`, `64^2`): the interface was 0.76 cells wide and the driving force
 * sat at 94% of the bistability ceiling. Neither was fixable alone — see
 * "a sub-grid interface needs a near-critical driving force" below.
 */
TEST_CASE("the shipped preset resolves its interface and keeps a margin",
          "[AllenCahn][kinetics][unit][preset]") {
  const allen_cahn::RunConfig cfg;
  constexpr double dx = 1.0;

  // 1. The interface spans enough cells for the discrete Laplacian to see it.
  const double width = allen_cahn::interface_width_cells(cfg.M, cfg.epsilon, dx);
  INFO("interface width = " << width << " cells");
  REQUIRE(width >= allen_cahn::RunConfig::kMinInterfaceWidthCells);

  // 2. The double well is still a double well, with room to spare.
  const double fraction =
      allen_cahn::driving_force_fraction(cfg.epsilon, cfg.driving_force);
  INFO("F eps^2 is at " << 100.0 * fraction << "% of the bistability ceiling");
  REQUIRE(fraction < 1.0);
  REQUIRE(fraction <= allen_cahn::RunConfig::kMaxDrivingForceFraction);

  // 3. The seed the app plants at the default grid is supercritical, so the
  //    demonstration demonstrates growth rather than dissolution. Buying
  //    margin in (2) costs exactly this: R* scales as 1/F.
  const double r0 = allen_cahn::seed_radius_cells(cfg.nx_glob, cfg.ny_glob);
  const double r_star =
      allen_cahn::critical_radius_cells(cfg.M, cfg.epsilon, cfg.driving_force, dx);
  INFO("seed radius " << r0 << " cells vs critical radius " << r_star);
  REQUIRE(r0 > 2.0 * r_star);

  // 4. Explicit Euler is stable with room to spare at the shipped dt.
  const double dt_diffusive = dx * dx / (4.0 * cfg.M);
  const double dt_reaction = cfg.epsilon * cfg.epsilon;
  REQUIRE(cfg.dt < 0.25 * std::min(dt_diffusive, dt_reaction));
}

TEST_CASE("a run too short to measure is skipped, not failed",
          "[AllenCahn][kinetics][unit]") {
  allen_cahn::RunConfig cfg;
  cfg.n_steps = 50;
  allen_cahn::AreaSamples a;
  a.initial = 52;
  a.half = 60;
  a.three_quarter = 70;
  a.final_ = 80;
  const auto k = allen_cahn::analyse_interface_kinetics(a, cfg, kDx);
  REQUIRE(k.verdict == allen_cahn::CheckVerdict::Skipped);
  REQUIRE(k.reason.find("shorter than") != std::string::npos);
}

TEST_CASE("a seed that does not grow fails the check",
          "[AllenCahn][kinetics][unit]") {
  // Explicit parameters rather than the shipped preset: this exercises the
  // analyser, and it should not have to be re-tuned every time the preset
  // moves. Fast front, so the curvature correction is small and the numbers
  // below stay readable: v_flat = 11.4, R* = M/v = 0.70 cells.
  allen_cahn::RunConfig cfg;
  cfg.n_steps = 5000;
  cfg.dt = 0.00009;
  cfg.M = 8.0;
  cfg.epsilon = 0.19;
  cfg.driving_force = 10.0;
  allen_cahn::AreaSamples a;
  a.initial = 52;
  a.half = 52;
  a.three_quarter = 52;
  a.final_ = 52;
  const auto k = allen_cahn::analyse_interface_kinetics(a, cfg, kDx);
  // The theory says this front should have advanced ~2.4 cells over the last
  // half of the run. It advanced none, so this is the physics failing, not
  // the run being too short.
  REQUIRE(k.v_predicted > 0.0);
  REQUIRE(k.verdict == allen_cahn::CheckVerdict::Fail);
  REQUIRE_THAT(k.v_late, WithinAbs(0.0, 1e-15));

  // The same standstill in a run whose *predicted* advance is sub-cell is
  // unmeasurable rather than wrong.
  allen_cahn::RunConfig weak = cfg;
  weak.driving_force = 0.01;
  const auto unmeasurable = allen_cahn::analyse_interface_kinetics(a, weak, kDx);
  REQUIRE(unmeasurable.verdict == allen_cahn::CheckVerdict::Skipped);

  // A front that advances steadily, by more than a cell, but at well under
  // the predicted speed is a genuine failure — not a measurement floor.
  a.half = 100;          // R = 5.642
  a.three_quarter = 121; // R = 6.206
  a.final_ = 144;        // R = 6.770, so dR = 1.13 cells over the last half
  const auto slow = allen_cahn::analyse_interface_kinetics(a, cfg, kDx);
  REQUIRE(slow.steady);
  REQUIRE(slow.r_final - slow.r_half > allen_cahn::RunConfig::kMinMeasurableAdvanceCells);
  REQUIRE(slow.v_late < allen_cahn::RunConfig::kVelocityBandLo * slow.v_predicted);
  REQUIRE(slow.verdict == allen_cahn::CheckVerdict::Fail);
  REQUIRE(slow.reason.find("outside the sharp-interface band") !=
          std::string::npos);
}

TEST_CASE("a subcritical seed is reported as dissolved, not as a slow front",
          "[AllenCahn][kinetics][unit]") {
  // Explicit parameters, so this keeps testing the analyser if the preset
  // moves again: M = 8, eps = 0.75, F = 0.25 gives v_flat = 1.125 and a
  // critical radius M/v = 7.11 cells.
  allen_cahn::RunConfig cfg;
  cfg.n_steps = 5000;
  cfg.dt = 0.005;
  cfg.M = 8.0;
  cfg.epsilon = 0.75;
  cfg.driving_force = 0.25;
  allen_cahn::AreaSamples a;
  a.initial = 52; // R0 = 4.07 cells, against a critical radius of 7.11
  a.half = 20;
  a.three_quarter = 4;
  a.final_ = 0;
  const auto k = allen_cahn::analyse_interface_kinetics(a, cfg, kDx);
  REQUIRE(k.r_initial < k.r_critical);
  REQUIRE(k.verdict == allen_cahn::CheckVerdict::Fail);
  REQUIRE(k.reason.find("below the critical radius") != std::string::npos);
}

TEST_CASE("an empty seed fails rather than dividing by zero",
          "[AllenCahn][kinetics][unit]") {
  allen_cahn::RunConfig cfg;
  cfg.n_steps = 5000;
  allen_cahn::AreaSamples a;
  const auto k = allen_cahn::analyse_interface_kinetics(a, cfg, kDx);
  REQUIRE(k.verdict == allen_cahn::CheckVerdict::Fail);
  REQUIRE(k.reason.find("N0 == 0") != std::string::npos);
}

/**
 * @brief The criterion has to describe the physics at every box size.
 *
 * Two grids, and the quantity that must agree is `v_late / v_predicted`, not
 * `v_late` itself: the seed scales with the box, so a bigger box measures a
 * bigger disc, and a bigger disc really does grow faster (`v = v_flat -
 * M/R`). Requiring the raw speeds to match would be requiring the physics to
 * be wrong.
 *
 * The old preset hid this. At `eps = 0.19, F = 10` the raw speeds at `64^2`
 * and `128^2` agree to 0.3% — but only because `M/R` was 6% of a very fast
 * front there; extend the same comparison to `512^2` and the raw speeds
 * spread by 10% while the residual against the flat law grows monotonically
 * from `+19%` to `+22%`. The pair the old test picked was the one pair that
 * happened to agree.
 */
TEST_CASE("the pass criterion is the same at 256^2 and 384^2",
          "[AllenCahn][kinetics][integration]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REQUIRE(nproc == 1);

  allen_cahn::RunConfig small; // shipped preset: 256^2
  allen_cahn::RunConfig large;
  large.nx_glob = 384;
  large.ny_glob = 384;

  const auto k_small =
      allen_cahn::analyse_interface_kinetics(run_and_sample(small), small, kDx);
  const auto k_large =
      allen_cahn::analyse_interface_kinetics(run_and_sample(large), large, kDx);

  std::cout << "interface velocity 256^2: " << k_small.v_late << " (predicted "
            << k_small.v_predicted << ")  384^2: " << k_large.v_late
            << " (predicted " << k_large.v_predicted
            << ")  flat-front law: " << k_small.v_theory << '\n';

  // The whole point: same physics, same verdict, whatever the box size.
  REQUIRE(k_small.verdict == allen_cahn::CheckVerdict::Pass);
  REQUIRE(k_large.verdict == allen_cahn::CheckVerdict::Pass);

  // Both grids sit on the curvature-corrected law to a few percent. Measured
  // on this machine: +1.6% and +1.3%.
  REQUIRE_THAT(k_small.v_late, WithinRel(k_small.v_predicted, 0.05));
  REQUIRE_THAT(k_large.v_late, WithinRel(k_large.v_predicted, 0.05));

  // ... and they agree with *each other* on how far off the law they are,
  // which the raw speeds (0.89 vs 0.95, 6.6% apart) do not.
  const double bias_small = k_small.v_late / k_small.v_predicted;
  const double bias_large = k_large.v_late / k_large.v_predicted;
  REQUIRE_THAT(bias_large, WithinRel(bias_small, 0.03));
}

/**
 * @brief Why the bistability margin could not be fixed on its own.
 *
 * The pre-0.2 preset sat 6% under the bistability ceiling. The obvious fix —
 * leave `eps = 0.19` and back the driving force off — does not work, and
 * this is the measurement that says so. At a 0.76-cell interface the lattice
 * pins the front: at the *same* margin fraction the shipped preset now uses,
 * a seed on a `64^2` box barely moves and the check fails, where the
 * resolved preset tracks the continuum law to 2%. The resolution had to be
 * fixed first, and fixing it is what made the margin affordable.
 */
TEST_CASE("a sub-grid interface needs a near-critical driving force",
          "[AllenCahn][kinetics][integration]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REQUIRE(nproc == 1);

  const allen_cahn::RunConfig shipped;
  const double fraction = allen_cahn::driving_force_fraction(
      shipped.epsilon, shipped.driving_force);

  allen_cahn::RunConfig subgrid; // the pre-0.2 geometry
  subgrid.nx_glob = 64;
  subgrid.ny_glob = 64;
  // 20000 steps, four times the shipped run: long enough that "the front did
  // not move" cannot be blamed on the run being short.
  subgrid.n_steps = 20000;
  subgrid.dt = 0.00009;
  subgrid.M = 8.0;
  subgrid.epsilon = 0.19;
  // Same distance from the ceiling as the shipped preset, at the old eps.
  subgrid.driving_force =
      fraction * allen_cahn::max_bistable_driving_force(subgrid.epsilon);

  REQUIRE(allen_cahn::interface_width_cells(subgrid.M, subgrid.epsilon, kDx) <
          allen_cahn::RunConfig::kMinInterfaceWidthCells);

  const auto k =
      allen_cahn::analyse_interface_kinetics(run_and_sample(subgrid), subgrid, kDx);
  std::cout << "sub-grid interface (0.76 cells) at the shipped margin: v_late="
            << k.v_late << " vs predicted " << k.v_predicted << '\n';

  // Not a pass. The front is there, it is just not moving at the speed the
  // continuum law says it should.
  REQUIRE(k.verdict != allen_cahn::CheckVerdict::Pass);
  REQUIRE(k.v_late < 0.75 * k.v_predicted);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
