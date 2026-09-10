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
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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
  // Shipped preset: M = 8, eps = 0.19, F = 10.
  REQUIRE_THAT(allen_cahn::sharp_interface_velocity(8.0, 0.19, 10.0),
               WithinRel(1.5 * 10.0 * 0.19 * 4.0, 1e-12));
  // Linear in the driving force, so no driving force means no front.
  REQUIRE_THAT(allen_cahn::sharp_interface_velocity(8.0, 0.19, 0.0),
               WithinAbs(0.0, 1e-15));
  // Interface half-width, in cells: below one the front is lattice-limited.
  REQUIRE_THAT(allen_cahn::interface_width_cells(8.0, 0.19, 1.0),
               WithinRel(0.76, 1e-9));
  // The shipped preset sits just under the bistability limit — 6% of margin.
  REQUIRE(10.0 < allen_cahn::max_bistable_driving_force(0.19));
  REQUIRE(10.0 > 0.9 * allen_cahn::max_bistable_driving_force(0.19));
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
  allen_cahn::RunConfig cfg;
  cfg.n_steps = 5000;
  allen_cahn::AreaSamples a;
  a.initial = 52;
  a.half = 52;
  a.three_quarter = 52;
  a.final_ = 52;
  const auto k = allen_cahn::analyse_interface_kinetics(a, cfg, kDx);
  // The theory says this front should have advanced ~2.6 cells over the last
  // half of the run. It advanced none, so this is the physics failing, not
  // the run being too short.
  REQUIRE(k.verdict == allen_cahn::CheckVerdict::Fail);
  REQUIRE_THAT(k.v_late, WithinAbs(0.0, 1e-15));

  // The same standstill in a run whose *predicted* advance is sub-cell is
  // unmeasurable rather than wrong.
  allen_cahn::RunConfig weak = cfg;
  weak.driving_force = 0.01;
  const auto unmeasurable = allen_cahn::analyse_interface_kinetics(a, weak, kDx);
  REQUIRE(unmeasurable.verdict == allen_cahn::CheckVerdict::Skipped);

  // A front that advances steadily, by more than a cell, but at well under
  // half the predicted speed is a genuine failure — not a measurement floor.
  a.half = 100;          // R = 5.642
  a.three_quarter = 121; // R = 6.206
  a.final_ = 144;        // R = 6.770, so dR = 1.13 cells over the last half
  const auto slow = allen_cahn::analyse_interface_kinetics(a, cfg, kDx);
  REQUIRE(slow.steady);
  REQUIRE(slow.r_final - slow.r_half > allen_cahn::RunConfig::kMinMeasurableAdvanceCells);
  REQUIRE(slow.v_late < allen_cahn::RunConfig::kVelocityBandLo * slow.v_theory);
  REQUIRE(slow.verdict == allen_cahn::CheckVerdict::Fail);
  REQUIRE(slow.reason.find("outside the sharp-interface band") !=
          std::string::npos);
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

TEST_CASE("the pass criterion is the same at 64^2 and 128^2",
          "[AllenCahn][kinetics][integration]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REQUIRE(nproc == 1);

  allen_cahn::RunConfig small;
  small.nx_glob = 64;
  small.ny_glob = 64;
  allen_cahn::RunConfig large;
  large.nx_glob = 128;
  large.ny_glob = 128;

  const auto k_small =
      allen_cahn::analyse_interface_kinetics(run_and_sample(small), small, kDx);
  const auto k_large =
      allen_cahn::analyse_interface_kinetics(run_and_sample(large), large, kDx);

  std::cout << "interface velocity 64^2: " << k_small.v_late
            << "  128^2: " << k_large.v_late
            << "  theory: " << k_small.v_theory << '\n';

  // The whole point: same physics, same verdict, whatever the box size.
  REQUIRE(k_small.verdict == allen_cahn::CheckVerdict::Pass);
  REQUIRE(k_large.verdict == allen_cahn::CheckVerdict::Pass);

  // And the measured speeds agree, which the area ratio never did (6.08 vs
  // 3.50 for these two grids).
  REQUIRE_THAT(k_large.v_late, WithinRel(k_small.v_late, 0.10));
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
