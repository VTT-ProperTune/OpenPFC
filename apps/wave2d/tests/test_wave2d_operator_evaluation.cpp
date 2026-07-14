// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_wave2d_operator_evaluation.cpp
 * @brief Catch2 unit tests for Wave2D operator-evaluation contract.
 *
 * @details
 * This test suite validates the operator-evaluation contract for the 2D wave
 * equation as a coupled first-order system, demonstrating:
 *
 * - Numerical equivalence between `WaveOperator::evaluate()` and the legacy
 *   `WaveModel::rhs()` method.
 * - Tuple protocol support for scattering coupled field increments.
 * - Non-mutation semantics (input state remains unchanged).
 * - Write-mode declarations (`overwrite` semantic).
 * - Metric scaling (inv_dx2, inv_dy2) application.
 *
 * Tests use 1e-12 tolerance for point-wise RHS comparisons, consistent with
 * the wave speed coupling precision and existing wave2d test patterns.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <iostream>

#include <wave2d/wave_model.hpp>
#include <wave2d/operator_evaluation.hpp>

using Catch::Matchers::WithinAbs;
using namespace wave2d;

TEST_CASE("WaveOperator::evaluate returns correct increments", "[wave2d][operator]") {
  const double inv_dx2 = 1.0 / 0.01;
  const double inv_dy2 = 1.0 / 0.02;
  WaveOperator op(inv_dx2, inv_dy2);
  WaveModel model{.inv_dx2 = inv_dx2, .inv_dy2 = inv_dy2};

  SECTION("zero v and zero Laplacian give zero increments") {
    WaveLaplacian lap{.lxx = 0.0, .lyy = 0.0};
    const double v_val = 0.0;
    const auto result = op.evaluate(lap, v_val, 0.0);
    const auto expected = model.rhs(0.0, v_val, lap);
    REQUIRE_THAT(result.increments.du, WithinAbs(expected.du, 1e-12));
    REQUIRE_THAT(result.increments.dv, WithinAbs(expected.dv, 1e-12));
    REQUIRE_THAT(result.increments.du, WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(result.increments.dv, WithinAbs(0.0, 1e-12));
  }

  SECTION("non-zero v and Laplacian match model.rhs") {
    WaveLaplacian lap{.lxx = 1.0, .lyy = 2.0};
    const double v_val = 0.5;
    const auto result = op.evaluate(lap, v_val, 1.0);
    const auto expected = model.rhs(1.0, v_val, lap);
    REQUIRE_THAT(result.increments.du, WithinAbs(expected.du, 1e-12));
    REQUIRE_THAT(result.increments.dv, WithinAbs(expected.dv, 1e-12));
  }

  SECTION("du equals input v value") {
    WaveLaplacian lap{.lxx = 0.0, .lyy = 0.0};
    const double v_val = 3.14159;
    const auto result = op.evaluate(lap, v_val, 0.0);
    REQUIRE_THAT(result.increments.du, WithinAbs(v_val, 1e-12));
    REQUIRE_THAT(result.increments.dv, WithinAbs(0.0, 1e-12));
  }

  SECTION("negative v produces negative du") {
    WaveLaplacian lap{.lxx = 0.0, .lyy = 0.0};
    const double v_val = -2.71828;
    const auto result = op.evaluate(lap, v_val, 0.0);
    REQUIRE_THAT(result.increments.du, WithinAbs(v_val, 1e-12));
    REQUIRE_THAT(result.increments.du, WithinAbs(-2.71828, 1e-12));
  }
}

TEST_CASE("WaveOperatorResult supports tuple protocol", "[wave2d][operator]") {
  WaveOperatorResult result{.increments = {.du = 1.5, .dv = -2.3}};
  auto [du, dv] = result.as_tuple();
  REQUIRE_THAT(du, WithinAbs(1.5, 1e-12));
  REQUIRE_THAT(dv, WithinAbs(-2.3, 1e-12));

  const auto& const_result = result;
  auto [c_du, c_dv] = const_result.as_tuple();
  REQUIRE_THAT(c_du, WithinAbs(1.5, 1e-12));
  REQUIRE_THAT(c_dv, WithinAbs(-2.3, 1e-12));
}

TEST_CASE("WaveOperator::evaluate does not modify input state", "[wave2d][operator]") {
  const double inv_dx2 = 1.0;
  const double inv_dy2 = 1.0;
  WaveOperator op(inv_dx2, inv_dy2);
  WaveLaplacian lap{.lxx = 1.0, .lyy = 2.0};
  const WaveLaplacian lap_copy = lap;

  (void)op.evaluate(lap, 0.5, 0.0);

  REQUIRE_THAT(lap.lxx, WithinAbs(lap_copy.lxx, 0.0));
  REQUIRE_THAT(lap.lyy, WithinAbs(lap_copy.lyy, 0.0));
}

TEST_CASE("WaveOperator applies metric scaling correctly", "[wave2d][operator]") {
  const double dx = 0.01;
  const double dy = 0.02;
  const double inv_dx2 = 1.0 / (dx * dx);
  const double inv_dy2 = 1.0 / (dy * dy);
  WaveOperator op(inv_dx2, inv_dy2);

  WaveLaplacian lap{.lxx = 1.0, .lyy = 1.0};
  const auto result = op.evaluate(lap, 0.0, 0.0);

  // dv = c² * (inv_dx²*lxx + inv_dy²*lyy)
  const double expected_lap_scaled = inv_dx2 + inv_dy2;
  REQUIRE_THAT(result.increments.dv, WithinAbs(kC * kC * expected_lap_scaled, 1e-12));
  REQUIRE_THAT(result.increments.du, WithinAbs(0.0, 1e-12));
}

TEST_CASE("WaveOperator declares overwrite write mode", "[wave2d][operator]") {
  REQUIRE(WaveOperator::write_mode == WriteMode::overwrite);
}

TEST_CASE("WaveOperator accepts time parameter", "[wave2d][operator]") {
  const double inv_dx2 = 1.0;
  const double inv_dy2 = 1.0;
  WaveOperator op(inv_dx2, inv_dy2);
  WaveModel model{.inv_dx2 = inv_dx2, .inv_dy2 = inv_dy2};

  WaveLaplacian lap{.lxx = 1.0, .lyy = 1.0};
  const double v_val = 0.5;
  const double t_arbitrary = 2.71828;

  const auto result = op.evaluate(lap, v_val, t_arbitrary);
  const auto expected = model.rhs(t_arbitrary, v_val, lap);

  REQUIRE_THAT(result.increments.du, WithinAbs(expected.du, 1e-12));
  REQUIRE_THAT(result.increments.dv, WithinAbs(expected.dv, 1e-12));
}

TEST_CASE("WaveOperator is constructible and callable", "[wave2d][operator]") {
  const double inv_dx2 = 1.0;
  const double inv_dy2 = 1.0;

  WaveOperator op1(inv_dx2, inv_dy2);
  REQUIRE(std::is_nothrow_constructible_v<WaveOperator, double, double>);

  WaveOperator op2(op1);
  REQUIRE(std::is_nothrow_copy_constructible_v<WaveOperator>);

  WaveOperator op3(std::move(op2));
  REQUIRE(std::is_nothrow_move_constructible_v<WaveOperator>);

  WaveLaplacian lap{.lxx = 0.5, .lyy = 0.5};
  const auto result = op1.evaluate(lap, 0.0, 0.0);
  REQUIRE_THAT(result.increments.du, WithinAbs(0.0, 1e-12));
}

int main(int argc, char* argv[]) {
  return Catch::Session().run(argc, argv);
}
