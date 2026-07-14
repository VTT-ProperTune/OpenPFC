// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_heat3d_operator_evaluation.cpp
 * @brief Catch2 unit tests for Heat3D operator-evaluation contract.
 *
 * @details
 * This test suite validates the operator-evaluation contract for the 3D heat
 * equation, demonstrating:
 *
 * - Numerical equivalence between `HeatOperator::evaluate()` and the legacy
 *   `HeatModel::rhs()` method.
 * - Non-mutation semantics (input state remains unchanged).
 * - Write-mode declarations (`overwrite` semantic).
 * - Stage context acceptance (time parameter handling).
 *
 * Tests use 1e-12 tolerance for point-wise RHS comparisons, consistent with
 * the thermal diffusion coefficient precision and existing heat3d test patterns.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <iostream>

#include <heat3d/heat_model.hpp>
#include <heat3d/operator_evaluation.hpp>

using Catch::Matchers::WithinAbs;
using namespace heat3d;

TEST_CASE("HeatOperator::evaluate returns correct RHS", "[heat3d][operator]") {
  HeatOperator op;
  HeatModel model;

  SECTION("zero Laplacian gives zero RHS") {
    HeatGrads g{.xx = 0.0, .yy = 0.0, .zz = 0.0};
    const auto result = op.evaluate(g, 0.0);
    const double expected = model.rhs(0.0, g);
    REQUIRE_THAT(result.d_u, WithinAbs(expected, 1e-12));
    REQUIRE_THAT(result.d_u, WithinAbs(0.0, 1e-12));
  }

  SECTION("uniform unit Laplacian gives RHS = 3 * kD") {
    HeatGrads g{.xx = 1.0, .yy = 1.0, .zz = 1.0};
    const auto result = op.evaluate(g, 0.0);
    const double expected = model.rhs(0.0, g);
    REQUIRE_THAT(result.d_u, WithinAbs(expected, 1e-12));
    REQUIRE_THAT(result.d_u, WithinAbs(3.0 * kD, 1e-12));
  }

  SECTION("arbitrary Laplacian matches model.rhs") {
    HeatGrads g{.xx = 1.0, .yy = -2.0, .zz = 0.5};
    const auto result = op.evaluate(g, 1.5);
    const double expected = model.rhs(1.5, g);
    REQUIRE_THAT(result.d_u, WithinAbs(expected, 1e-12));
  }

  SECTION("negative Laplacian produces negative RHS") {
    HeatGrads g{.xx = -1.0, .yy = -2.0, .zz = -3.0};
    const auto result = op.evaluate(g, 0.0);
    const double expected = model.rhs(0.0, g);
    const double lap = -1.0 - 2.0 - 3.0; // -6.0
    REQUIRE_THAT(result.d_u, WithinAbs(expected, 1e-12));
    REQUIRE_THAT(result.d_u, WithinAbs(kD * lap, 1e-12));
  }
}

TEST_CASE("HeatOperator declares overwrite write mode", "[heat3d][operator]") {
  REQUIRE(HeatOperator::write_mode == WriteMode::overwrite);
}

TEST_CASE("HeatOperator::evaluate does not modify input state", "[heat3d][operator]") {
  HeatOperator op;
  HeatGrads g{.xx = 1.0, .yy = 2.0, .zz = 3.0};
  const HeatGrads g_copy = g;

  (void)op.evaluate(g, 0.0);

  REQUIRE_THAT(g.xx, WithinAbs(g_copy.xx, 0.0));
  REQUIRE_THAT(g.yy, WithinAbs(g_copy.yy, 0.0));
  REQUIRE_THAT(g.zz, WithinAbs(g_copy.zz, 0.0));
}

TEST_CASE("HeatOperator::evaluate accepts time parameter", "[heat3d][operator]") {
  HeatOperator op;
  HeatModel model;
  HeatGrads g{.xx = 1.0, .yy = 1.0, .zz = 1.0};

  const double t_arbitrary = 3.14159;
  const auto result = op.evaluate(g, t_arbitrary);
  const double expected = model.rhs(t_arbitrary, g);

  REQUIRE_THAT(result.d_u, WithinAbs(expected, 1e-12));
}

TEST_CASE("HeatOperator is constructible and callable", "[heat3d][operator]") {
  HeatOperator op;
  REQUIRE(std::is_nothrow_default_constructible_v<HeatOperator>);
  REQUIRE(std::is_nothrow_copy_constructible_v<HeatOperator>);
  REQUIRE(std::is_nothrow_move_constructible_v<HeatOperator>);

  HeatGrads g{.xx = 0.5, .yy = 0.5, .zz = 0.5};
  const auto result = op.evaluate(g, 0.0);
  REQUIRE_THAT(result.d_u, WithinAbs(1.5 * kD, 1e-12));
}

int main(int argc, char* argv[]) {
  return Catch::Session().run(argc, argv);
}
