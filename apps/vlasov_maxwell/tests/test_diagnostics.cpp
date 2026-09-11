// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_diagnostics.cpp
 * @brief The estimators every quoted rate and amplitude passes through.
 *
 * @details
 * These are not helpers; they are the measurement apparatus, and three of
 * the four bugs found during the first validation campaign were in here
 * rather than in the physics. A biased estimator produces a number that is
 * wrong while every conservation diagnostic reads `1e-14`, which is the
 * hardest kind of error to notice and the easiest kind to publish.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

#include <mpi.h>

#include <vlasov_maxwell/diagnostics.hpp>

using Catch::Approx;

namespace {
constexpr double kPi = 3.14159265358979323846;
} // namespace

TEST_CASE("mode_amplitude returns the amplitude, not N times it",
          "[diagnostics]") {
  // A cos(k_m x) sampled at cell centres. The DFT here is unnormalised and
  // a real signal splits between +m and -m, so recovering A needs both a
  // 1/N and a factor 2. An earlier revision had only the 2, which made
  // every reported field strength N times too large -- invisibly, because
  // a constant factor cancels out of every growth rate.
  for (const int nx : {16, 32, 64, 128}) {
    const double Lx = 2.0 * kPi;
    vlasov::SpectralLine1D line(nx, Lx);
    for (const int m : {1, 2, 5}) {
      if (m >= nx / 2) continue;
      for (const double A : {1.0, 1.0e-4, 3.7}) {
        std::vector<double> g(static_cast<std::size_t>(nx));
        for (int i = 0; i < nx; ++i) {
          const double x = (static_cast<double>(i) + 0.5) * Lx / nx;
          g[static_cast<std::size_t>(i)] = A * std::cos(m * 2.0 * kPi * x / Lx);
        }
        INFO("nx=" << nx << " m=" << m << " A=" << A);
        REQUIRE(vlasov::mode_amplitude(line, g, m) == Approx(A).epsilon(1e-12));
        // and a mode that is not there must read zero
        if (m + 1 < nx / 2) {
          REQUIRE(vlasov::mode_amplitude(line, g, m + 1) ==
                  Approx(0.0).margin(1e-12 * A));
        }
      }
    }
  }
}

TEST_CASE("the envelope fit is unbiased where the plain fit is not",
          "[diagnostics]") {
  // The exact situation that cost -9.9% on the Landau rate: a damped
  // oscillation sampled over a window that is NOT a whole number of
  // half-periods. Both estimators see the same data; only one of them
  // returns the rate that was put in.
  const double gamma = -0.1533594669;
  const double omega = 1.4156618886;
  std::vector<double> t, y;
  for (int i = 0; i <= 2000; ++i) {
    const double ti = 0.02 * i;
    t.push_back(ti);
    y.push_back(std::exp(gamma * ti) * std::fabs(std::cos(omega * ti + 0.3)));
  }
  // A window of 16.0 spans 16/(pi/omega) = 7.21 half-periods: not integral.
  const double bad = vlasov::fit_exponential_rate(t, y, 2.0, 18.0);
  const double good = vlasov::fit_envelope_rate(t, y, 2.0, 18.0);
  INFO("plain " << bad << "  envelope " << good << "  true " << gamma);
  REQUIRE(good == Approx(gamma).epsilon(2e-3));
  // and the plain fit really is biased, by percents -- this is an assertion
  // about the estimator, so that if someone "fixes" it the test says why.
  REQUIRE(std::fabs(bad - gamma) > 20.0 * std::fabs(good - gamma));
}

TEST_CASE("frequency_from_minima recovers the oscillation frequency",
          "[diagnostics]") {
  const double gamma = -0.15;
  const double omega = 1.41566;
  std::vector<double> t, y;
  for (int i = 0; i <= 4000; ++i) {
    const double ti = 0.01 * i;
    t.push_back(ti);
    y.push_back(std::exp(gamma * ti) * std::fabs(std::cos(omega * ti)));
  }
  REQUIRE(vlasov::frequency_from_minima(t, y, 0.0, 30.0) ==
          Approx(omega).epsilon(1e-3));
}

TEST_CASE("auto_growth_window excludes the saturation it is shown",
          "[diagnostics]") {
  // Exponential growth to t = 40, then a hard plateau -- the two-stream
  // shape. A window fixed as "the first half" would average the two and
  // report a rate low by a third; the data-chosen window must not.
  const double g = 0.3;
  std::vector<double> t, y;
  for (int i = 0; i <= 1500; ++i) {
    const double ti = 0.1 * i;
    t.push_back(ti);
    y.push_back(std::fmin(1.0e-5 * std::exp(g * ti), 1.0));
  }
  const auto w = vlasov::auto_growth_window(t, y);
  REQUIRE(std::isfinite(w[0]));
  REQUIRE(std::isfinite(w[1]));
  REQUIRE(w[1] < 40.0); // strictly inside the exponential phase
  REQUIRE(vlasov::fit_exponential_rate(t, y, w[0], w[1]) ==
          Approx(g).epsilon(1e-6));
  // A series that never grows must yield no window rather than a rate.
  std::vector<double> flat(t.size(), 1.0e-5);
  const auto none = vlasov::auto_growth_window(t, flat);
  REQUIRE(!std::isfinite(none[0]));
}

TEST_CASE("fit_rotation_rate gets the sign and flags aliasing",
          "[diagnostics]") {
  // The gyro oracle. d/dt (v_x + i v_y) = -i w (v_x + i v_y) makes the
  // phase advance at -w, and an earlier revision compared the measurement
  // against +w and reported 200% error on a magnitude that was right to
  // seven digits.
  for (const double w : {0.5, -0.5, 1.0}) {
    std::vector<double> t, px, py;
    for (int i = 0; i <= 800; ++i) {
      const double ti = 0.05 * i;
      t.push_back(ti);
      px.push_back(std::cos(w * ti));
      py.push_back(std::sin(w * ti));
    }
    bool aliased = true;
    INFO("w = " << w);
    REQUIRE(vlasov::fit_rotation_rate(t, px, py, &aliased) ==
            Approx(w).epsilon(1e-9));
    REQUIRE(!aliased);
  }
  // Sampled slower than half a turn per sample: must be flagged, because a
  // silent wrap reports a plausible slower rate.
  std::vector<double> t, px, py;
  for (int i = 0; i <= 200; ++i) {
    const double ti = 3.0 * i;
    t.push_back(ti);
    px.push_back(std::cos(1.0 * ti));
    py.push_back(std::sin(1.0 * ti));
  }
  bool aliased = false;
  (void)vlasov::fit_rotation_rate(t, px, py, &aliased);
  REQUIRE(aliased);
}

int main(int argc, char *argv[]) {
  // MPI is initialised even though nothing here is distributed: the
  // headers under test construct types that query MPI when it is up, and a
  // suite that only works outside an MPI context is a suite that will not
  // run the same way as the application it is testing.
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
