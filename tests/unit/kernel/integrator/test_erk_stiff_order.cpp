// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Checked comparison of Zoto & Bowman Table 10 (ERK43ZB) with the
 * Hochbruck–Ostermann stiff-order conditions. The test calls the shipped
 * coefficient and residual functions. It does not step an ODE.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/integrator/erk_stiff_order.hpp>
#include <openpfc/kernel/integrator/matrix_phi_2x2.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>

#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>

using Catch::Matchers::WithinAbs;
using pfc::integrator::evaluate_exponential_rk;
using pfc::integrator::ExponentialRkEvaluation;
using pfc::integrator::ExponentialRkMethod;
using pfc::integrator::hochbruck_ostermann_residuals;
using pfc::integrator::hochbruck_ostermann_weak_order4_residuals;
using pfc::integrator::Matrix2x2;
using pfc::integrator::matrix_phi;
using pfc::integrator::phi_function;
using pfc::integrator::spectral_exp_coeffs;
using pfc::integrator::StiffOrderResiduals;

namespace {

const Matrix2x2 kArgument{-1.5, 0.4, 0.2, -0.7};
const Matrix2x2 kJacobian{0.0, 1.0, 0.0, 0.0};
const Matrix2x2 kMixed{0.0, 0.0, 1.0, 0.3};
const Matrix2x2 kZero{};

[[nodiscard]] double frobenius(const Matrix2x2 &matrix) {
  return std::sqrt(matrix.m00 * matrix.m00 + matrix.m01 * matrix.m01 +
                   matrix.m10 * matrix.m10 + matrix.m11 * matrix.m11);
}

[[nodiscard]] StiffOrderResiduals residuals_of(ExponentialRkMethod method,
                                               const Matrix2x2 &argument) {
  return hochbruck_ostermann_residuals(evaluate_exponential_rk(method, argument),
                                       argument, kJacobian, kMixed);
}

void require_below(const StiffOrderResiduals &residuals,
                   std::initializer_list<int> conditions, double tolerance) {
  for (const int condition : conditions) {
    REQUIRE(residuals.condition[condition - 1] < tolerance);
  }
}

void require_above(const StiffOrderResiduals &residuals,
                   std::initializer_list<int> conditions, double floor) {
  for (const int condition : conditions) {
    REQUIRE(residuals.condition[condition - 1] > floor);
  }
}

} // namespace

TEST_CASE("phi_function matches the recurrence and the 2x2 phi through order 2",
          "[integrator][erk_stiff_order]") {
  for (int order = 0; order <= 4; ++order) {
    double factorial = 1.0;
    for (int i = 1; i <= order; ++i) {
      factorial *= static_cast<double>(i);
    }
    const Matrix2x2 at_zero = phi_function(kZero, order);
    REQUIRE_THAT(at_zero.m00, WithinAbs(1.0 / factorial, 1e-15));
    REQUIRE_THAT(at_zero.m11, WithinAbs(1.0 / factorial, 1e-15));
    REQUIRE_THAT(at_zero.m01, WithinAbs(0.0, 0.0));
    REQUIRE_THAT(at_zero.m10, WithinAbs(0.0, 0.0));
  }

  constexpr double z = 1.7;
  const Matrix2x2 diagonal{z, 0.0, 0.0, z};
  const double phi0 = std::exp(z);
  const double phi1 = (phi0 - 1.0) / z;
  const double phi2 = (phi1 - 1.0) / z;
  const double phi3 = (phi2 - 0.5) / z;
  const double phi4 = (phi3 - (1.0 / 6.0)) / z;
  const double expected[5] = {phi0, phi1, phi2, phi3, phi4};
  for (int order = 0; order <= 4; ++order) {
    const Matrix2x2 value = phi_function(diagonal, order);
    REQUIRE_THAT(value.m00, WithinAbs(expected[order], 1e-12));
    REQUIRE_THAT(value.m11, WithinAbs(expected[order], 1e-12));
  }

  const auto known = matrix_phi(kArgument);
  REQUIRE(frobenius(known.phi0 + (-1.0 * phi_function(kArgument, 0))) < 1e-12);
  REQUIRE(frobenius(known.phi1 + (-1.0 * phi_function(kArgument, 1))) < 1e-12);
  REQUIRE(frobenius(known.phi2 + (-1.0 * phi_function(kArgument, 2))) < 1e-12);

  constexpr double rate = -4.0;
  constexpr double step = 0.25;
  const auto coeffs = spectral_exp_coeffs(rate, step);
  const Matrix2x2 symbol{rate * step, 0.0, 0.0, rate * step};
  REQUIRE_THAT(step * phi_function(symbol, 1).m00, WithinAbs(coeffs.phi1_L, 1e-12));
}

TEST_CASE("ERK4K is classical RK4 at zero and misses stiff conditions 6, 7 and 8",
          "[integrator][erk_stiff_order]") {
  const Matrix2x2 commutator =
      kArgument * kJacobian + (-1.0 * (kJacobian * kArgument));
  REQUIRE(frobenius(commutator) > 0.1);

  const ExponentialRkEvaluation classical =
      evaluate_exponential_rk(ExponentialRkMethod::erk4k, kZero);
  REQUIRE(classical.stage_count == 4);
  REQUIRE_THAT(classical.weight[1][0].m00, WithinAbs(0.5, 1e-15));
  REQUIRE_THAT(classical.weight[2][0].m00, WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(classical.weight[2][1].m00, WithinAbs(0.5, 1e-15));
  REQUIRE_THAT(classical.weight[3][0].m00, WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(classical.weight[3][1].m00, WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(classical.weight[3][2].m00, WithinAbs(1.0, 1e-15));
  REQUIRE_THAT(classical.output[0].m00, WithinAbs(1.0 / 6.0, 1e-15));
  REQUIRE_THAT(classical.output[1].m00, WithinAbs(1.0 / 3.0, 1e-15));
  REQUIRE_THAT(classical.output[2].m00, WithinAbs(1.0 / 3.0, 1e-15));
  REQUIRE_THAT(classical.output[3].m00, WithinAbs(1.0 / 6.0, 1e-15));
  require_below(residuals_of(ExponentialRkMethod::erk4k, kZero),
                {1, 2, 3, 4, 5, 6, 7, 8, 9}, 1e-12);

  const StiffOrderResiduals stiff =
      residuals_of(ExponentialRkMethod::erk4k, kArgument);
  require_below(stiff, {1, 2, 3, 4, 5, 9}, 1e-12);
  require_above(stiff, {6, 7, 8}, 1e-4);
  REQUIRE(stiff.condition8_repeated_index > 1e-4);
}

TEST_CASE("ERK43ZB order-4 row meets stiff conditions 1-5 and misses 6-9",
          "[integrator][erk_stiff_order]") {
  const ExponentialRkEvaluation method =
      evaluate_exponential_rk(ExponentialRkMethod::erk43zb_order4, kArgument);
  REQUIRE(method.stage_count == 5);
  REQUIRE_THAT(method.node[1], WithinAbs(1.0 / 6.0, 0.0));
  REQUIRE_THAT(method.node[2], WithinAbs(0.5, 0.0));
  REQUIRE_THAT(method.node[4], WithinAbs(1.0, 0.0));

  require_below(residuals_of(ExponentialRkMethod::erk43zb_order4, kZero),
                {1, 2, 3, 4, 5, 6, 7, 8, 9}, 1e-12);

  const StiffOrderResiduals stiff =
      residuals_of(ExponentialRkMethod::erk43zb_order4, kArgument);
  require_below(stiff, {1, 2, 3, 4, 5}, 1e-12);
  require_above(stiff, {6, 7, 9}, 1e-4);
  REQUIRE(stiff.condition[7] > 1e-5);
  REQUIRE(stiff.condition8_repeated_index > 1e-4);
}

TEST_CASE("ERK43ZB embedded row stays below order 4 at a zero argument",
          "[integrator][erk_stiff_order]") {
  const ExponentialRkEvaluation method =
      evaluate_exponential_rk(ExponentialRkMethod::erk43zb_embedded, kZero);
  REQUIRE(method.stage_count == 4);
  const StiffOrderResiduals at_zero =
      residuals_of(ExponentialRkMethod::erk43zb_embedded, kZero);
  require_below(at_zero, {1, 2, 3, 4, 5, 8}, 1e-12);
  REQUIRE(at_zero.condition[5] > 1e-2);
  REQUIRE(at_zero.condition[6] > 1e-3);
  REQUIRE(at_zero.condition[8] > 1e-3);

  const StiffOrderResiduals stiff =
      residuals_of(ExponentialRkMethod::erk43zb_embedded, kArgument);
  require_below(stiff, {1, 2, 3}, 1e-12);
  require_above(stiff, {4, 5, 6}, 1e-3);
}

TEST_CASE("ERK43ZB satisfies the weak fourth-order fixed-step conditions",
          "[integrator][erk_stiff_order]") {
  // Independent Taylor-series evaluation of the same published tableau
  // at three materially different non-commuting/nonnormal operators was
  // checked before freezing these tests; do not silently replace Z by 0.
  const Matrix2x2 alternative{-0.5, 1.2, -0.1, -1.2};
  const Matrix2x2 defective{0.0, 2.0, 0.0, 0.0};
  const Matrix2x2 triangular{-2.0, 0.0, 0.6, -0.05};
  const auto origin =
      evaluate_exponential_rk(ExponentialRkMethod::erk43zb_order4, kZero);
  const std::array<Matrix2x2, 4> examples{kArgument, alternative, defective,
                                          triangular};
  for (const auto &argument : examples) {
    const auto candidate =
        evaluate_exponential_rk(ExponentialRkMethod::erk43zb_order4, argument);
    const auto strong =
        hochbruck_ostermann_residuals(candidate, argument, kJacobian, kMixed);
    require_below(strong, {1, 2, 3, 4, 5}, 2e-11);
    const auto weak = hochbruck_ostermann_weak_order4_residuals(
        candidate, origin, argument, kJacobian, kMixed);
    REQUIRE(weak.psi4_at_zero < 2e-11);
    REQUIRE(weak.condition7 < 2e-11);
    REQUIRE(weak.condition8 < 2e-11);
    REQUIRE(weak.condition9 < 2e-11);
  }
  // Output weights matter to the theorem; altering the origin values
  // must be observable by a nonzero-Z weak-order test.
  auto corrupted = origin;
  const Matrix2x2 corruption{0.1, 0.0, 0.0, 0.1};
  corrupted.output[3] = corrupted.output[3] + corruption;
  const auto candidate =
      evaluate_exponential_rk(ExponentialRkMethod::erk43zb_order4, kArgument);
  const auto weak_corrupted = hochbruck_ostermann_weak_order4_residuals(
      candidate, corrupted, kArgument, kJacobian, kMixed);
  REQUIRE(weak_corrupted.psi4_at_zero > 1e-4);
  REQUIRE(weak_corrupted.condition7 > 1e-5);
}

TEST_CASE("phi-function rejects non-finite and unscalable arguments",
          "[integrator][erk_stiff_order]") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const Matrix2x2 bad{nan, 0.0, 0.0, 0.0};
  REQUIRE_THROWS_AS(phi_function(bad, 2), std::invalid_argument);
  const Matrix2x2 extreme{std::numeric_limits<double>::max(), 0.0, 0.0, 0.0};
  REQUIRE_THROWS_AS(phi_function(extreme, 1), std::out_of_range);
}

TEST_CASE("changing an ERK output weight breaks stiff condition 1",
          "[integrator][erk_stiff_order]") {
  ExponentialRkEvaluation method =
      evaluate_exponential_rk(ExponentialRkMethod::erk43zb_order4, kArgument);
  const StiffOrderResiduals original =
      hochbruck_ostermann_residuals(method, kArgument, kJacobian, kMixed);
  REQUIRE(original.condition[0] < 1e-12);

  method.output[0] = method.output[0] + Matrix2x2{1.0, 0.0, 0.0, 1.0};
  const StiffOrderResiduals broken =
      hochbruck_ostermann_residuals(method, kArgument, kJacobian, kMixed);
  REQUIRE(broken.condition[0] > 0.5);
}
