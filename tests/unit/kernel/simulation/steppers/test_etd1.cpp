// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <span>
#include <tuple>
#include <vector>

#include <openpfc/kernel/integrator/etd1_apply.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/kernel/simulation/steppers/etd1.hpp>

using pfc::integrator::fill_spectral_exp_coeffs;
using pfc::integrator::spectral_exp_coeffs;
using pfc::sim::steppers::ETD1Stepper;
using pfc::sim::steppers::MultiETD1Stepper;

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

  ETD1Stepper stepper(dt, 1, rhs);
  stepper.set_coefficients(exp_buf, phi_buf);

  std::vector<double> u{u0};
  double t = 0.0;
  for (std::size_t s = 0; s < n_steps; ++s) {
    auto attempt = stepper.attempt(t, u);
    REQUIRE(attempt.success);
    u[0] = stepper.candidate()[0];
    t = attempt.t1;
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
  ETD1Stepper stepper(dt, 1, rhs);
  stepper.set_coefficients(exp_buf, phi_buf);

  std::vector<double> u{u0};
  auto attempt = stepper.attempt(0.0, u);
  REQUIRE(attempt.success);
  REQUIRE(attempt.t1 == Catch::Approx(dt));

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
  ETD1Stepper stepper(dt, Lvals.size(), rhs);
  stepper.set_coefficients(exp_buf, phi_buf);

  std::vector<double> u(Lvals.size(), 2.0);
  auto attempt = stepper.attempt(0.0, u);
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
  ETD1Stepper stepper(dt, 2, rhs);
  stepper.set_coefficients(exp_buf, phi_buf);

  std::vector<double> u{1.0, 2.0};
  const std::vector<double> fingerprint = u;

  auto ok = stepper.attempt(0.0, u);
  REQUIRE(ok.success);
  REQUIRE(u == fingerprint);

  // Mismatched coeff size: set equal-length wrong-sized spans, then attempt.
  std::vector<double> bad_exp{1.0};
  std::vector<double> bad_phi{0.1};
  stepper.set_coefficients(bad_exp, bad_phi);
  auto fail = stepper.attempt(0.0, u);
  REQUIRE_FALSE(fail.success);
  REQUIRE(u == fingerprint);

  // Wrong-sized accepted vector with good coeffs.
  stepper.set_coefficients(exp_buf, phi_buf);
  std::vector<double> wrong_u{1.0};
  auto fail2 = stepper.attempt(0.0, wrong_u);
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
  MultiETD1Stepper<TwoFieldConstantN, 2> stepper(dt, {1, 1}, rhs);
  stepper.set_coefficients({std::span<const double>{exp0},
                            std::span<const double>{exp1}},
                           {std::span<const double>{phi0},
                            std::span<const double>{phi1}});

  std::vector<double> u0{2.0};
  std::vector<double> u1{4.0};
  const auto fp0 = u0;
  const auto fp1 = u1;

  auto attempt = stepper.attempt(0.0, u0, u1);
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

struct ThreeFieldConstantN {
  double n0{0.0};
  double n1{0.0};
  double n2{0.0};
  void operator()(
      double /*t*/,
      std::tuple<std::vector<double> &, std::vector<double> &,
                 std::vector<double> &> /*u_pack*/,
      std::tuple<std::vector<double> &, std::vector<double> &,
                 std::vector<double> &>
          du_pack) const {
    std::get<0>(du_pack)[0] = n0;
    std::get<1>(du_pack)[0] = n1;
    std::get<2>(du_pack)[0] = n2;
  }
};

TEST_CASE("etd1_three_field_bundle", "[stepper][etd1]") {
  constexpr double dt = 0.1;
  std::vector<double> L0{-1.0}, L1{-2.0}, L2{-4.0};
  std::vector<double> exp0(1), phi0(1), exp1(1), phi1(1), exp2(1), phi2(1);
  fill_spectral_exp_coeffs(L0, dt, exp0, phi0);
  fill_spectral_exp_coeffs(L1, dt, exp1, phi1);
  fill_spectral_exp_coeffs(L2, dt, exp2, phi2);

  ThreeFieldConstantN rhs{0.25, -0.5, 1.0};
  MultiETD1Stepper<ThreeFieldConstantN, 3> stepper(dt, {1, 1, 1}, rhs);
  stepper.set_coefficients(
      {std::span<const double>{exp0}, std::span<const double>{exp1},
       std::span<const double>{exp2}},
      {std::span<const double>{phi0}, std::span<const double>{phi1},
       std::span<const double>{phi2}});

  std::vector<double> u0{1.0}, u1{2.0}, u2{3.0};
  const auto fp0 = u0;
  const auto fp1 = u1;
  const auto fp2 = u2;
  auto attempt = stepper.attempt(0.0, u0, u1, u2);
  REQUIRE(attempt.success);
  REQUIRE(u0 == fp0);
  REQUIRE(u1 == fp1);
  REQUIRE(u2 == fp2);

  const auto r0 = spectral_exp_coeffs(L0[0], dt);
  const auto r1 = spectral_exp_coeffs(L1[0], dt);
  const auto r2 = spectral_exp_coeffs(L2[0], dt);
  REQUIRE(stepper.candidate(0)[0] ==
          Catch::Approx(r0.exp_Ldt * 1.0 + r0.phi1_L * 0.25).margin(1e-12));
  REQUIRE(stepper.candidate(1)[0] ==
          Catch::Approx(r1.exp_Ldt * 2.0 + r1.phi1_L * (-0.5)).margin(1e-12));
  REQUIRE(stepper.candidate(2)[0] ==
          Catch::Approx(r2.exp_Ldt * 3.0 + r2.phi1_L * 1.0).margin(1e-12));
}

using Complex = std::complex<double>;

struct ZeroComplexN {
  void operator()(double /*t*/, std::vector<Complex> & /*u*/,
                  std::vector<Complex> &du) const {
    for (auto &d : du) {
      d = Complex{0.0, 0.0};
    }
  }
};

struct ConstantComplexN {
  Complex value{};
  void operator()(double /*t*/, std::vector<Complex> & /*u*/,
                  std::vector<Complex> &du) const {
    for (auto &d : du) {
      d = value;
    }
  }
};

TEST_CASE("etd1_complex_stiff_linear_exact", "[stepper][etd1][complex]") {
  // u' = L u with real stiff L; N = 0. ETD1 is exact: u(t) = exp(L t) u0.
  constexpr double L = -80.0;
  constexpr double dt = 0.05;
  constexpr Complex u0{1.25, -0.4};

  std::vector<double> Lvec{L};
  std::vector<double> exp_buf(1);
  std::vector<double> phi_buf(1);
  fill_spectral_exp_coeffs(Lvec, dt, exp_buf, phi_buf);

  ZeroComplexN rhs{};
  ETD1Stepper<ZeroComplexN, Complex> stepper(dt, 1, rhs);
  stepper.set_coefficients(exp_buf, phi_buf);

  std::vector<Complex> u{u0};
  const auto fingerprint = u;
  auto attempt = stepper.attempt(0.0, u);
  REQUIRE(attempt.success);
  REQUIRE(u == fingerprint);

  const Complex expected = std::exp(Complex{L * dt, 0.0}) * u0;
  REQUIRE(stepper.candidate()[0].real() ==
          Catch::Approx(expected.real()).margin(1e-12));
  REQUIRE(stepper.candidate()[0].imag() ==
          Catch::Approx(expected.imag()).margin(1e-12));

  pfc::sim::steppers::commit_step_attempt(u, attempt);
  REQUIRE(u[0].real() == Catch::Approx(expected.real()).margin(1e-12));
}

TEST_CASE("etd1_complex_closed_form_with_N", "[stepper][etd1][complex]") {
  constexpr double L = -12.0;
  constexpr double dt = 0.1;
  constexpr Complex u0{0.5, 0.25};
  constexpr Complex Nval{0.1, -0.2};

  std::vector<double> Lvec{L};
  std::vector<double> exp_buf(1);
  std::vector<double> phi_buf(1);
  fill_spectral_exp_coeffs(Lvec, dt, exp_buf, phi_buf);

  ConstantComplexN rhs{Nval};
  ETD1Stepper<ConstantComplexN, Complex> stepper(dt, 1, rhs);
  stepper.set_coefficients(exp_buf, phi_buf);

  std::vector<Complex> u{u0};
  auto attempt = stepper.attempt(0.0, u);
  REQUIRE(attempt.success);

  const auto ref = spectral_exp_coeffs(L, dt);
  const Complex expected = Complex(ref.exp_Ldt) * u0 + Complex(ref.phi1_L) * Nval;
  REQUIRE(stepper.candidate()[0].real() ==
          Catch::Approx(expected.real()).margin(1e-12));
  REQUIRE(stepper.candidate()[0].imag() ==
          Catch::Approx(expected.imag()).margin(1e-12));
}

struct TwoFieldConstantComplexN {
  Complex n0{};
  Complex n1{};
  void operator()(
      double /*t*/,
      std::tuple<std::vector<Complex> &, std::vector<Complex> &> /*u_pack*/,
      std::tuple<std::vector<Complex> &, std::vector<Complex> &> du_pack)
      const {
    std::get<0>(du_pack)[0] = n0;
    std::get<1>(du_pack)[0] = n1;
  }
};

TEST_CASE("etd1_multi_field_complex_bundle", "[stepper][etd1][complex]") {
  constexpr double dt = 0.1;
  std::vector<double> L0{-8.0};
  std::vector<double> L1{-20.0};
  std::vector<double> exp0(1), phi0(1), exp1(1), phi1(1);
  fill_spectral_exp_coeffs(L0, dt, exp0, phi0);
  fill_spectral_exp_coeffs(L1, dt, exp1, phi1);

  const Complex u0_val{0.5, -0.25};
  const Complex u1_val{-1.0, 0.75};
  const Complex n0{0.1, 0.2};
  const Complex n1{-0.3, 0.05};

  TwoFieldConstantComplexN rhs{n0, n1};
  MultiETD1Stepper<TwoFieldConstantComplexN, 2, Complex> stepper(dt, {1, 1},
                                                                 rhs);
  stepper.set_coefficients({std::span<const double>{exp0},
                            std::span<const double>{exp1}},
                           {std::span<const double>{phi0},
                            std::span<const double>{phi1}});

  std::vector<Complex> u0{u0_val};
  std::vector<Complex> u1{u1_val};
  const auto fp0 = u0;
  const auto fp1 = u1;

  auto attempt = stepper.attempt(0.0, u0, u1);
  REQUIRE(attempt.success);
  REQUIRE(u0 == fp0);
  REQUIRE(u1 == fp1);

  const auto r0 = spectral_exp_coeffs(L0[0], dt);
  const auto r1 = spectral_exp_coeffs(L1[0], dt);
  const Complex e0 = Complex(r0.exp_Ldt) * u0_val + Complex(r0.phi1_L) * n0;
  const Complex e1 = Complex(r1.exp_Ldt) * u1_val + Complex(r1.phi1_L) * n1;
  REQUIRE(stepper.candidate(0)[0].real() ==
          Catch::Approx(e0.real()).margin(1e-12));
  REQUIRE(stepper.candidate(0)[0].imag() ==
          Catch::Approx(e0.imag()).margin(1e-12));
  REQUIRE(stepper.candidate(1)[0].real() ==
          Catch::Approx(e1.real()).margin(1e-12));
  REQUIRE(stepper.candidate(1)[0].imag() ==
          Catch::Approx(e1.imag()).margin(1e-12));

  pfc::sim::steppers::commit_step_attempt(u0, u1, attempt);
  REQUIRE(u0[0].real() == Catch::Approx(e0.real()).margin(1e-12));
  REQUIRE(u1[0].imag() == Catch::Approx(e1.imag()).margin(1e-12));
}

TEST_CASE("apply_etd1_update host real and complex", "[integrator][etd1_apply]") {
  using pfc::integrator::apply_etd1_update;
  using Complex = std::complex<double>;

  const std::vector<double> exp_Ldt{0.5, 2.0};
  const std::vector<double> phi1_L{0.25, -0.1};
  std::vector<double> u{1.0, -2.0};
  std::vector<double> nlin{4.0, 8.0};
  std::vector<double> out(2, 0.0);
  apply_etd1_update(std::span<const double>{exp_Ldt},
                    std::span<const double>{phi1_L},
                    std::span<const double>{u}, std::span<const double>{nlin},
                    std::span<double>{out});
  REQUIRE(out[0] == Catch::Approx(0.5 * 1.0 + 0.25 * 4.0).margin(1e-14));
  REQUIRE(out[1] == Catch::Approx(2.0 * -2.0 + -0.1 * 8.0).margin(1e-14));

  std::vector<Complex> uc{Complex{1.0, 0.5}, Complex{0.0, -1.0}};
  std::vector<Complex> nc{Complex{0.2, -0.1}, Complex{1.0, 0.0}};
  std::vector<Complex> oc(2);
  apply_etd1_update(std::span<const double>{exp_Ldt},
                    std::span<const double>{phi1_L},
                    std::span<const Complex>{uc}, std::span<const Complex>{nc},
                    std::span<Complex>{oc});
  const Complex e0 = Complex(exp_Ldt[0]) * uc[0] + Complex(phi1_L[0]) * nc[0];
  REQUIRE(oc[0].real() == Catch::Approx(e0.real()).margin(1e-14));
  REQUIRE(oc[0].imag() == Catch::Approx(e0.imag()).margin(1e-14));
}
