// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <span>
#include <tuple>
#include <vector>

#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/kernel/simulation/steppers/etd1.hpp>

using pfc::integrator::fill_spectral_exp_coeffs;
using pfc::integrator::spectral_exp_coeffs;
using pfc::sim::steppers::Etd1Stepper;
using pfc::sim::steppers::MultiEtd1Stepper;

namespace {

struct ConstantN {
  double value{0.0};
  void operator()(double /*t*/, std::vector<double> & /*u*/,
                  std::vector<double> &du) const {
    for (double &d : du) {
      d = value;
    }
  }
};

struct MutatingN {
  void operator()(double /*t*/, std::vector<double> &u,
                  std::vector<double> &du) const {
    for (std::size_t i = 0; i < u.size(); ++i) {
      du[i] = 1.0;
      u[i] = -999.0; // must not affect caller's accepted buffer
    }
  }
};

struct QuadraticN {
  void operator()(double /*t*/, std::vector<double> &u,
                  std::vector<double> &du) const {
    for (std::size_t i = 0; i < u.size(); ++i) {
      du[i] = u[i] * u[i];
    }
  }
};

struct TwoFieldConstantN {
  double n0{0.0};
  double n1{0.0};
  void operator()(double /*t*/,
                  std::tuple<std::vector<double> &, std::vector<double> &> u_pack,
                  std::tuple<std::vector<double> &, std::vector<double> &>
                      du_pack) const {
    auto &u0 = std::get<0>(u_pack);
    auto &u1 = std::get<1>(u_pack);
    auto &du0 = std::get<0>(du_pack);
    auto &du1 = std::get<1>(du_pack);
    for (std::size_t i = 0; i < du0.size(); ++i) {
      du0[i] = n0;
      u0[i] = -111.0;
    }
    for (std::size_t i = 0; i < du1.size(); ++i) {
      du1[i] = n1;
      u1[i] = -222.0;
    }
  }
};

[[nodiscard]] std::vector<double>
integrate_etd1(double L, QuadraticN rhs, double u0, double T, double dt) {
  const std::size_t n_steps = static_cast<std::size_t>(std::llround(T / dt));
  std::vector<double> Lvec{L};
  std::vector<double> exp_buf(1);
  std::vector<double> phi_buf(1);
  fill_spectral_exp_coeffs(Lvec, dt, exp_buf, phi_buf);

  Etd1Stepper stepper(dt, 1, rhs);
  stepper.set_coefficients(exp_buf, phi_buf);

  std::vector<double> u{u0};
  double t = 0.0;
  for (std::size_t s = 0; s < n_steps; ++s) {
    auto attempt = stepper.attempt_step(t, u);
    REQUIRE(attempt.success);
    u[0] = stepper.candidate()[0];
    t = attempt.t_next;
  }
  return u;
}

} // namespace

TEST_CASE("etd1_closed_form_diagonal_update", "[stepper][etd1]") {
  constexpr double dt = 0.1;
  constexpr double L = -2.5;
  constexpr double u0 = 1.25;
  constexpr double Nval = 0.4;

  std::vector<double> Lvec{L};
  std::vector<double> exp_buf(1);
  std::vector<double> phi_buf(1);
  fill_spectral_exp_coeffs(Lvec, dt, exp_buf, phi_buf);

  ConstantN rhs{Nval};
  Etd1Stepper stepper(dt, 1, rhs);
  stepper.set_coefficients(exp_buf, phi_buf);

  std::vector<double> u{u0};
  auto attempt = stepper.attempt_step(0.0, u);
  REQUIRE(attempt.success);
  REQUIRE(attempt.t_next == Catch::Approx(dt));

  const auto ref = spectral_exp_coeffs(L, dt);
  const double expected = ref.exp_Ldt * u0 + ref.phi1_L * Nval;
  REQUIRE(stepper.candidate()[0] == Catch::Approx(expected).margin(1e-12));
  REQUIRE(u[0] == Catch::Approx(u0)); // accepted untouched
}

TEST_CASE("etd1_near_zero_phi1_finite", "[stepper][etd1]") {
  constexpr double dt = 0.05;
  const std::vector<double> Lvals{0.0, 1e-14, -5e-13};

  std::vector<double> exp_buf(Lvals.size());
  std::vector<double> phi_buf(Lvals.size());
  fill_spectral_exp_coeffs(Lvals, dt, exp_buf, phi_buf);

  ConstantN rhs{1.0};
  Etd1Stepper stepper(dt, Lvals.size(), rhs);
  stepper.set_coefficients(exp_buf, phi_buf);

  std::vector<double> u(Lvals.size(), 2.0);
  auto attempt = stepper.attempt_step(0.0, u);
  REQUIRE(attempt.success);

  for (std::size_t i = 0; i < Lvals.size(); ++i) {
    REQUIRE(std::isfinite(stepper.candidate()[i]));
    const double taylor_phi1 = dt + 0.5 * Lvals[i] * dt * dt;
    REQUIRE(phi_buf[i] == Catch::Approx(taylor_phi1).margin(1e-15));
    const double expected = exp_buf[i] * u[i] + phi_buf[i] * 1.0;
    REQUIRE(stepper.candidate()[i] == Catch::Approx(expected).margin(1e-12));
  }
}

TEST_CASE("etd1_accepted_state_isolation", "[stepper][etd1]") {
  constexpr double dt = 0.1;
  std::vector<double> Lvec{-1.0, -2.0};
  std::vector<double> exp_buf(2);
  std::vector<double> phi_buf(2);
  fill_spectral_exp_coeffs(Lvec, dt, exp_buf, phi_buf);

  MutatingN rhs;
  Etd1Stepper stepper(dt, 2, rhs);
  stepper.set_coefficients(exp_buf, phi_buf);

  std::vector<double> u{1.0, 2.0};
  const std::vector<double> fingerprint = u;

  auto ok = stepper.attempt_step(0.0, u);
  REQUIRE(ok.success);
  REQUIRE(u == fingerprint);

  // Mismatched coeff size: set equal-length wrong-sized spans, then attempt.
  std::vector<double> bad_exp{1.0};
  std::vector<double> bad_phi{0.1};
  stepper.set_coefficients(bad_exp, bad_phi);
  auto fail = stepper.attempt_step(0.0, u);
  REQUIRE_FALSE(fail.success);
  REQUIRE(u == fingerprint);

  // Wrong-sized accepted vector with good coeffs.
  stepper.set_coefficients(exp_buf, phi_buf);
  std::vector<double> wrong_u{1.0};
  auto fail2 = stepper.attempt_step(0.0, wrong_u);
  REQUIRE_FALSE(fail2.success);
  REQUIRE(u == fingerprint);
}

TEST_CASE("etd1_first_order_temporal_convergence", "[stepper][etd1]") {
  // Manufactured diagonal ODE: u' = L*u + u^2 with L < 0.
  // Exact solution for reference via fine ETD1 (same method, tiny dt).
  constexpr double L = -1.5;
  constexpr double u0 = 0.5;
  constexpr double T = 0.4;
  QuadraticN rhs;

  const auto fine = integrate_etd1(L, rhs, u0, T, T / 512.0);
  const auto coarse = integrate_etd1(L, rhs, u0, T, T / 32.0);
  const auto mid = integrate_etd1(L, rhs, u0, T, T / 64.0);

  const double e_coarse = std::fabs(coarse[0] - fine[0]);
  const double e_mid = std::fabs(mid[0] - fine[0]);
  REQUIRE(e_coarse > 0.0);
  REQUIRE(e_mid > 0.0);
  const double ratio = e_coarse / e_mid;
  // First-order: halving dt ≈ halves error → ratio ≈ 2; allow [1.5, 2.5].
  REQUIRE(ratio == Catch::Approx(2.0).margin(0.5));
}

TEST_CASE("etd1_multi_field_bundle", "[stepper][etd1]") {
  constexpr double dt = 0.1;
  std::vector<double> L0{-1.0};
  std::vector<double> L1{-3.0};
  std::vector<double> exp0(1), phi0(1), exp1(1), phi1(1);
  fill_spectral_exp_coeffs(L0, dt, exp0, phi0);
  fill_spectral_exp_coeffs(L1, dt, exp1, phi1);

  TwoFieldConstantN rhs{0.5, -0.25};
  MultiEtd1Stepper<TwoFieldConstantN, 2> stepper(dt, {1, 1}, rhs);
  stepper.set_coefficients({std::span<const double>{exp0},
                            std::span<const double>{exp1}},
                           {std::span<const double>{phi0},
                            std::span<const double>{phi1}});

  std::vector<double> u0{2.0};
  std::vector<double> u1{4.0};
  const auto fp0 = u0;
  const auto fp1 = u1;

  auto attempt = stepper.attempt_step(0.0, u0, u1);
  REQUIRE(attempt.success);
  REQUIRE(u0 == fp0);
  REQUIRE(u1 == fp1);

  const auto r0 = spectral_exp_coeffs(L0[0], dt);
  const auto r1 = spectral_exp_coeffs(L1[0], dt);
  REQUIRE(stepper.candidate(0)[0] ==
          Catch::Approx(r0.exp_Ldt * 2.0 + r0.phi1_L * 0.5).margin(1e-12));
  REQUIRE(stepper.candidate(1)[0] ==
          Catch::Approx(r1.exp_Ldt * 4.0 + r1.phi1_L * (-0.25)).margin(1e-12));
}
