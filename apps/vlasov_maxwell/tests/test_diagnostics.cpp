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
#include <vlasov_maxwell/ics.hpp>

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

TEST_CASE("the envelope estimator refuses a monotone exponential",
          "[diagnostics]") {
  // Weibel magnetic energy is a growing exponential, not a damped
  // oscillation. Fitting its peaks is the Landau estimator applied to the
  // wrong signal: a strictly increasing series has no interior maxima, so
  // the honest answer is NaN rather than a rate from a single endpoint.
  const double g = 0.35;
  std::vector<double> t, y;
  for (int i = 0; i <= 400; ++i) {
    const double ti = 0.05 * i;
    t.push_back(ti);
    y.push_back(1.0e-5 * std::exp(g * ti));
  }
  REQUIRE_FALSE(std::isfinite(vlasov::fit_envelope_rate(t, y, 0.0, 20.0)));
  REQUIRE(vlasov::fit_exponential_rate(t, y, 0.0, 20.0) == Approx(g).epsilon(1e-9));
}

TEST_CASE("frequency_from_minima works on |E|^2 and refuses a short window",
          "[diagnostics]") {
  // The driver comment talks about minima of the *energy*. Those sit at the
  // same times as the minima of |E|, so the estimator must recover omega
  // from either series. A window covering fewer than two minima must not
  // invent a period.
  const double gamma = -0.15;
  const double omega = 1.41566;
  std::vector<double> t, amp, energy;
  for (int i = 0; i <= 4000; ++i) {
    const double ti = 0.01 * i;
    t.push_back(ti);
    const double a = std::exp(gamma * ti) * std::fabs(std::cos(omega * ti));
    amp.push_back(a);
    energy.push_back(a * a);
  }
  REQUIRE(vlasov::frequency_from_minima(t, energy, 0.0, 30.0) ==
          Approx(omega).epsilon(1e-3));
  REQUIRE_FALSE(std::isfinite(vlasov::frequency_from_minima(t, energy, 0.0, 0.5)));
}

TEST_CASE("auto_growth_window excludes logistic saturation", "[diagnostics]") {
  // A hard plateau (the two-stream shape already tested) is the easy case.
  // A logistic rolls over smoothly, which is what a weakly saturating
  // instability looks like, and a window that includes the roll-off is the
  // bias that cost -13.5% at ceiling = 0.2 on the two-stream run. The
  // driver therefore uses ceiling = 0.05; that choice is the measurement.
  const double g = 0.3;
  const double ymax = 1.0;
  const double y0 = 1.0e-5;
  std::vector<double> t, y;
  for (int i = 0; i <= 2000; ++i) {
    const double ti = 0.05 * i;
    t.push_back(ti);
    y.push_back(ymax / (1.0 + (ymax / y0 - 1.0) * std::exp(-g * ti)));
  }
  const auto tight = vlasov::auto_growth_window(t, y, 5.0, 0.05);
  REQUIRE(std::isfinite(tight[0]));
  REQUIRE(std::isfinite(tight[1]));
  const double good = vlasov::fit_exponential_rate(t, y, tight[0], tight[1]);
  INFO("tight window [" << tight[0] << ", " << tight[1] << "]  rate " << good);
  // Instantaneous logistic rate at y = 0.05 ymax is 0.95 g; the window
  // average sits between g and that. Measured 0.291 against 0.3.
  REQUIRE(good == Approx(g).epsilon(0.05));

  const double whole = vlasov::fit_exponential_rate(t, y, 0.0, t.back());
  REQUIRE(std::fabs(whole - g) > 5.0 * std::fabs(good - g));

  // The looser default ceiling includes the roll-off. That is why the
  // two-stream driver does not use it; lock the failure in so a later
  // "simplification" that drops the 0.05 cannot pass as an improvement.
  const auto loose = vlasov::auto_growth_window(t, y, 5.0, 0.2);
  const double biased = vlasov::fit_exponential_rate(t, y, loose[0], loose[1]);
  REQUIRE(std::fabs(biased - g) > std::fabs(good - g));
}

TEST_CASE("fitting total energy of a two-mode field picks the wrong mode",
          "[diagnostics]") {
  // The Weibel campaign: a seeded mode growing at g1 and a quieter parasite
  // growing at g2 > g1. Total magnetic energy is the sum of both, so after
  // crossover the energy slope is 2 g2 -- which is how a growth rate came
  // out 79% high while every conservation diagnostic read 1e-14. The
  // observable linear theory predicts is |hat B(m_seed)|.
  const double g1 = 0.20;
  const double g2 = 0.50;
  const double A1_0 = 1.0e-3;
  const double A2_0 = 1.0e-6;
  const int nx = 64;
  const double Lx = 2.0 * kPi;
  vlasov::SpectralLine1D line(nx, Lx);

  std::vector<double> t, e_tot, m1;
  for (int n = 0; n <= 400; ++n) {
    const double ti = 0.1 * static_cast<double>(n);
    t.push_back(ti);
    const double a1 = A1_0 * std::exp(g1 * ti);
    const double a2 = A2_0 * std::exp(g2 * ti);
    std::vector<double> Bz(static_cast<std::size_t>(nx));
    for (int i = 0; i < nx; ++i) {
      const double x = (static_cast<double>(i) + 0.5) * Lx / nx;
      Bz[static_cast<std::size_t>(i)] = a1 * std::cos(x) + a2 * std::cos(2.0 * x);
    }
    e_tot.push_back(vlasov::component_energy(line, Bz));
    m1.push_back(vlasov::mode_amplitude(line, Bz, 1));
  }

  const auto w_mode = vlasov::auto_growth_window(t, m1);
  REQUIRE(std::isfinite(w_mode[0]));
  const double g_mode = vlasov::fit_exponential_rate(t, m1, w_mode[0], w_mode[1]);
  INFO("mode-1 rate " << g_mode << "  true " << g1);
  REQUIRE(g_mode == Approx(g1).epsilon(2e-3));

  // After crossover (t ~ ln(A1_0/A2_0)/(g2-g1) = 23) energy is the parasite.
  // Amplitude energy ~ A^2 so the slope is 2 g, not g. Measured 0.999 against
  // 1.0 over [28, 40]; 1% leaves room for the leftover seeded mode.
  const double g_energy_late = vlasov::fit_exponential_rate(t, e_tot, 28.0, 40.0);
  INFO("late energy rate " << g_energy_late << "  2 g2 = " << (2.0 * g2));
  REQUIRE(g_energy_late == Approx(2.0 * g2).epsilon(0.02));
  REQUIRE(std::fabs(g_energy_late / 2.0 - g1) > 0.2);
}

TEST_CASE("relative Gauss is a 0/0 for a Weibel-neutral plasma",
          "[diagnostics]") {
  // A Weibel plasma stays charge-neutral to ~1e-5 while the field is ~8e-2.
  // gauss_residual divides by max|rho|, so the relative form reads O(1)
  // while the unmatched charge is 1e-4 of the field. The campaign number
  // was 0.27 relative against an absolute residual of 4e-6. The ledger
  // has to carry field_scale and the absolute residual so a reader can
  // form the ratio that means something; this test is that they do.
  vlasov::SimParams p;
  p.nx = 64;
  p.Lx = 2.0 * kPi;
  vlasov::SpectralLine1D line(p.nx, p.Lx);
  const double B = 8.0e-2;
  const double rho_amp = 1.0e-5;

  vlasov::FieldState f = vlasov::FieldState::zeros(p.nx);
  std::vector<double> rho(static_cast<std::size_t>(p.nx));
  for (int i = 0; i < p.nx; ++i) {
    const double x = p.x_of(i);
    f.Ey[static_cast<std::size_t>(i)] = B * std::cos(x);
    f.Bz[static_cast<std::size_t>(i)] = B * std::cos(x);
    f.Ex[static_cast<std::size_t>(i)] = 0.0;
    rho[static_cast<std::size_t>(i)] = rho_amp * std::cos(x);
  }
  const auto g = vlasov::gauss_residual(line, f.Ex, rho);
  double rho_max = 0.0;
  for (double v : rho) rho_max = std::fmax(rho_max, std::fabs(v));
  // Cell centres of a 64-point grid miss the cosine peak by cos(pi/64),
  // so rho_max is 0.9988 * rho_amp, not rho_amp. Measured 9.988e-6.
  REQUIRE(rho_max == Approx(rho_amp * std::cos(kPi / p.nx)).epsilon(1e-12));
  // Ex = 0, so |d_x Ex - rho| / max|rho| = 1: Gauss looks 100% broken.
  REQUIRE(g.residual == Approx(1.0).epsilon(1e-9));
  REQUIRE(g.abs_residual == Approx(rho_max).epsilon(1e-12));

  auto src = vlasov::Sources::zeros(p.nx);
  src.rho = rho;
  const vlasov::Ledger L =
      vlasov::make_ledger(p, line, {}, src, f, g, 0.0, 0, 1);
  REQUIRE(L.field_scale == Approx(B * std::cos(kPi / p.nx)).epsilon(1e-12));
  REQUIRE(L.gauss_residual == Approx(1.0).epsilon(1e-9));
  REQUIRE(L.gauss_abs_residual == Approx(rho_max).epsilon(1e-12));
  // Relative Gauss is O(1); the unmatched charge is 1e-4 of the field.
  // Measured rho_max / field_scale = 1.25e-4.
  REQUIRE(L.gauss_abs_residual / L.field_scale < 2.0e-4);
  REQUIRE(L.gauss_residual > 0.5);
  REQUIRE(L.energy_bz == Approx(0.25 * p.Lx * B * B).epsilon(1e-12));
}

TEST_CASE("drift of a vanishing baseline is absolute, not a 0/0",
          "[diagnostics]") {
  // Total momentum starts at zero in every benchmark. A relative drift
  // would be (now - 0)/0. The convention is the absolute change, and
  // set_drifts has to honour it or a healthy run reports inf.
  REQUIRE(vlasov::drift(1.0e-8, 0.0) == Approx(1.0e-8));
  REQUIRE(vlasov::drift(2.0, 1.0) == Approx(1.0));
  vlasov::Ledger ref{};
  vlasov::Ledger now{};
  ref.number = 1.0;
  now.number = 1.0 + 1.0e-12;
  ref.total_energy = 0.5;
  now.total_energy = 0.5 * (1.0 + 2.0e-14);
  ref.total_momentum_x = 0.0;
  now.total_momentum_x = 3.0e-9;
  vlasov::set_drifts(now, ref);
  // (1 + 1e-12) - 1 is not a binary 1e-12; compare against the arithmetic
  // the estimator actually saw. Margin 1e-18 is ~0.1 ulp of the inputs.
  REQUIRE(now.d_number ==
          Approx((now.number - ref.number) / ref.number).margin(1e-18));
  REQUIRE(now.d_energy ==
          Approx((now.total_energy - ref.total_energy) / ref.total_energy)
              .margin(1e-18));
  REQUIRE(now.d_momentum_x == Approx(3.0e-9).margin(0.0));
}

TEST_CASE("require_resolved_spacing refuses the Poisson-summation grid",
          "[diagnostics]") {
  // A Landau box with v_max = 8 v_th and nvy = 8 has dv = 2 v_th, i.e.
  // half a cell per thermal width. The midpoint sum of the Maxwellian is
  // then wrong by 2 exp(-2 pi^2 (v_th/dv)^2) = 1.44%, which is exactly
  // the density defect the first campaign deposited and which a scan over
  // n_vx cannot see. The gate has to refuse that grid, and has to accept
  // four cells per v_th, where the same formula is 1e-137.
  const double vth = 0.05;
  const double coarse = 2.0 * vth; // 0.5 cells / v_th
  const double fine = vth / 4.0;   // 4 cells / v_th
  const double pred = 2.0 * std::exp(-2.0 * kPi * kPi * 0.25);
  REQUIRE(vlasov::ics::maxwellian_quadrature_error(coarse, vth) ==
          Approx(pred).epsilon(1e-12));
  REQUIRE(pred == Approx(1.44e-2).epsilon(0.02));
  REQUIRE_THROWS_AS(vlasov::ics::require_resolved_spacing(coarse, vth, 1.0e-10),
                    std::invalid_argument);
  REQUIRE_NOTHROW(vlasov::ics::require_resolved_spacing(fine, vth, 1.0e-10));
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
