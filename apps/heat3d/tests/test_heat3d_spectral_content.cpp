// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_heat3d_spectral_content.cpp
 * @brief The closed-form half of the spectral-vs-FD accuracy study, pinned.
 *
 * @details
 * `heat3d_spectral_content_study` takes a couple of minutes because it also
 * runs real simulations, and CI does not run it. Everything it is *built
 * on* is instant arithmetic, so that is checked here:
 *
 *  - the FD symbol reproduced from the shipped `EvenCentralD2` tables;
 *  - the arcsin-series identity those tables turn out to satisfy exactly,
 *    which is what lets the dispersion defect be computed without
 *    catastrophic cancellation (see `heat3d/spectral_content_study.hpp`);
 *  - the defect being positive, and matching the direct difference where
 *    the direct difference is still trustworthy;
 *  - the error map's design orders, and its independence of the mode-cube
 *    size — the property that makes the map a statement about the field's
 *    spectral content rather than about one grid;
 *  - the crossover predicate, which is the single number the report
 *    quotes, together with the measured costs it is fed.
 *
 * Deliberately *not* checked here: the semi-analytic map against real
 * runs. That is `run_validation()`, which needs minutes of RK4, and lives
 * in the driver.
 */

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

#include <mpi.h>

#include <heat3d/spectral_content_study.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
namespace sc = heat3d::spectral_content;

TEST_CASE("heat3d spectral content: the FD symbol is the stencil's own",
          "[heat3d][spectral-content][unit]") {
  // Order 2 has a symbol everyone knows by heart: -4 sin^2(theta/2). The
  // tolerance is 1e-12 rather than round-off because `fd_symbol` evaluates
  // the stencil sum as written -- at theta = 0.1 that is 2cos(0.1) - 2,
  // which loses two digits to cancellation. Which is the whole reason the
  // *defect* is computed by a different route; see the next test.
  for (double theta : {0.1, 0.7, 1.5, 3.0}) {
    const double s = std::sin(0.5 * theta);
    REQUIRE_THAT(sc::fd_symbol(2, theta), WithinRel(-4.0 * s * s, 1e-12));
  }
  // Every tabulated order is consistent (symbol -> 0 as theta -> 0) and
  // second-order accurate at leading order (symbol -> -theta^2).
  for (int order : {2, 4, 6, 8, 10, 12, 20}) {
    REQUIRE_THAT(sc::fd_symbol(order, 0.0), WithinAbs(0.0, 1e-12));
    const double theta = 1e-3;
    REQUIRE_THAT(sc::fd_symbol(order, theta), WithinRel(-theta * theta, 1e-6));
  }
  REQUIRE_THROWS(sc::fd_symbol(3, 0.5));
}

TEST_CASE("heat3d spectral content: the shipped stencils ARE the truncated "
          "arcsin series",
          "[heat3d][spectral-content][unit]") {
  // (2 arcsin(delta/2))^2 = sum_m a_m delta^{2m} with delta^2 = 2-2cos(theta).
  // The claim -- the whole reason the defect can be computed as a
  // positive-term tail -- is that the order-2M table is that series
  // truncated at m = M. If a future edit to fd_stencils.hpp broke it, the
  // map would silently lose its precision guarantee, so pin it.
  for (int order : {2, 4, 6, 8, 10, 12, 20}) {
    const int m_max = order / 2;
    for (double theta : {0.2, 0.8, 1.6, 2.4, 3.0}) {
      const double delta_sq = 2.0 - 2.0 * std::cos(theta);
      double series = 0.0;
      double power = 1.0;
      for (int m = 1; m <= m_max; ++m) {
        power *= delta_sq;
        series += sc::arcsin_series_coefficient(m) * power;
      }
      INFO("order " << order << " theta " << theta);
      REQUIRE_THAT(sc::fd_symbol(order, theta), WithinAbs(-series, 1e-13));
    }
  }
  // Spot-check the coefficients themselves against the closed form.
  REQUIRE_THAT(sc::arcsin_series_coefficient(1), WithinRel(1.0, 1e-15));
  REQUIRE_THAT(sc::arcsin_series_coefficient(2), WithinRel(1.0 / 12.0, 1e-15));
  REQUIRE_THAT(sc::arcsin_series_coefficient(3), WithinRel(1.0 / 90.0, 1e-15));
  // The full series converges to theta^2 -- the identity itself.
  {
    const double theta = 1.0;
    const double delta_sq = 2.0 - 2.0 * std::cos(theta);
    double series = 0.0, power = 1.0;
    for (int m = 1; m <= 200; ++m) {
      power *= delta_sq;
      series += sc::arcsin_series_coefficient(m) * power;
    }
    REQUIRE_THAT(series, WithinRel(theta * theta, 1e-14));
  }
}

TEST_CASE("heat3d spectral content: the dispersion defect is positive and "
          "matches the direct difference where that still works",
          "[heat3d][spectral-content][unit]") {
  for (int order : {2, 4, 6, 8, 12}) {
    for (double theta : {0.4, 0.9, 1.4, 2.0, 2.8}) {
      const double d = sc::fd_symbol_defect(order, theta);
      INFO("order " << order << " theta " << theta << " defect " << d);
      // The stencil always under-resolves curvature: |lambda_fd| < k^2.
      REQUIRE(d > 0.0);
      const double direct = theta * theta + sc::fd_symbol(order, theta);
      // Loosest where the two terms cancel hardest (high order, small
      // theta); 1e-5 relative is far tighter than the direct form deserves
      // at order 12 / theta = 0.4 and still passes.
      REQUIRE_THAT(direct, WithinRel(d, 1e-5));
    }
  }
  // Where the direct difference has nothing left, the series still does.
  // Order 12 at theta = 0.1: the true defect is ~1.2e-19, three orders of
  // magnitude below what a double subtraction of two O(0.01) numbers can
  // represent.
  const double tiny = sc::fd_symbol_defect(12, 0.1);
  REQUIRE(tiny > 0.0);
  REQUIRE(tiny < 1e-18);
  REQUIRE(tiny > 1e-20);
  // Leading order: defect ~ a_{M+1} * theta^{2M+2}, so halving theta
  // divides it by 2^{2M+2}. Order 4 (M=2): factor 64.
  const double d1 = sc::fd_symbol_defect(4, 0.02);
  const double d2 = sc::fd_symbol_defect(4, 0.01);
  REQUIRE_THAT(d1 / d2, WithinRel(64.0, 1e-3));
}

TEST_CASE("heat3d spectral content: the error map recovers each design order",
          "[heat3d][spectral-content][unit]") {
  // At small f the map's error must scale as f^p: halving the content
  // fraction (i.e. doubling the grid at fixed physics) divides the error
  // by 2^p. This is the broadband analogue of what
  // test_heat3d_fd_convergence.cpp checks for one mode, and it is the
  // property the whole cost comparison rests on.
  for (int order : {2, 4, 6, 8}) {
    const double e1 = sc::predict_l2_error(order, 0.20, sc::auto_map_grid(0.20));
    const double e2 = sc::predict_l2_error(order, 0.10, sc::auto_map_grid(0.10));
    const double p = std::log(e1 / e2) / std::log(2.0);
    INFO("order " << order << ": " << e1 << " -> " << e2 << ", observed " << p);
    REQUIRE(p > order - 0.4);
    REQUIRE(p < order + 0.4);
  }
  // Monotone in f: a coarser grid is never more accurate. The bisection in
  // content_fraction_at() assumes this.
  for (int order : {2, 8}) {
    double previous = 0.0;
    for (double f : {0.2, 0.3, 0.4, 0.5, 0.6, 0.8}) {
      const double e = sc::predict_l2_error(order, f, sc::auto_map_grid(f));
      REQUIRE(e > previous);
      previous = e;
    }
  }
}

TEST_CASE("heat3d spectral content: the map does not depend on the mode cube",
          "[heat3d][spectral-content][unit]") {
  // This is what makes "content reaching fraction f of Nyquist" the right
  // parameter: the answer is a property of the field's spectrum relative to
  // the grid, not of the grid. It holds once k_c = f*N/2 is large enough to
  // sample the Gaussian spectrum, which N = 48 already is at f = 0.3.
  for (int order : {2, 4, 8, 12}) {
    const double reference = sc::predict_l2_error(order, 0.3, 128);
    for (int N : {48, 64, 96}) {
      INFO("order " << order << " N " << N);
      REQUIRE_THAT(sc::predict_l2_error(order, 0.3, N), WithinRel(reference, 1e-6));
    }
  }
}

TEST_CASE("heat3d spectral content: the crossover predicate",
          "[heat3d][spectral-content][unit]") {
  // Measured per-step costs at N=1024 on 8 GCDs, from
  // docs/report/data/heat3d_method_cost.csv. The crossover is the cube root
  // of the cost ratio and nothing else: under an N^3 cost model a 31.8x
  // per-step advantage buys only 3.17x in grid spacing.
  constexpr double kSpectral = 213.82;
  const std::vector<std::pair<int, double>> kFd = {
      {2, 6.73}, {4, 11.46}, {6, 15.01}, {8, 18.64}, {12, 24.69}};

  REQUIRE_THAT(sc::crossover_fraction(kFd[0].second, kSpectral),
               WithinRel(0.31573465, 1e-6));
  REQUIRE_THAT(sc::crossover_fraction(kFd[4].second, kSpectral),
               WithinRel(0.48695735, 1e-6));
  // Every order's threshold lands in a narrow band around a third to a
  // half of Nyquist, which is the report's headline.
  for (const auto &[order, cost] : kFd) {
    const double thresh = sc::crossover_fraction(cost, kSpectral);
    INFO("FD-" << order << " crossover fraction " << thresh);
    REQUIRE(thresh > 0.30);
    REQUIRE(thresh < 0.50);
  }
  // Consistency of the two spellings of the same statement.
  for (const auto &[order, cost] : kFd) {
    const double thresh = sc::crossover_fraction(cost, kSpectral);
    CHECK(sc::fd_cheaper_than_spectral(thresh * 1.01, cost, kSpectral));
    CHECK_FALSE(sc::fd_cheaper_than_spectral(thresh * 0.99, cost, kSpectral));
    CHECK_THAT(sc::equal_accuracy_cost_ratio(thresh, cost, kSpectral),
               WithinRel(1.0, 1e-12));
  }
  REQUIRE_THROWS(sc::crossover_fraction(-1.0, kSpectral));
}

TEST_CASE("heat3d spectral content: FD-12 ties the spectral path near 1e-6",
          "[heat3d][spectral-content][unit]") {
  // The chapter's headline number. Below this target no finite-difference
  // order in the shipped table is the cheaper way to get there.
  constexpr double kSpectral = 213.82;
  const double thresh = sc::crossover_fraction(24.69, kSpectral);
  const double eps_tie = sc::predict_l2_error(12, thresh, sc::auto_map_grid(thresh));
  INFO("FD-12 ties spectral at L2 = " << eps_tie);
  REQUIRE(eps_tie > 3e-7);
  REQUIRE(eps_tie < 1.5e-6);

  // And the corollary: at a 1e-4 target FD-12 is still several times
  // cheaper, at 1e-8 it is several times dearer.
  const double f_loose = sc::content_fraction_at(12, 1e-4);
  const double f_tight = sc::content_fraction_at(12, 1e-8);
  CHECK(sc::fd_cheaper_than_spectral(f_loose, 24.69, kSpectral));
  CHECK_FALSE(sc::fd_cheaper_than_spectral(f_tight, 24.69, kSpectral));
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
