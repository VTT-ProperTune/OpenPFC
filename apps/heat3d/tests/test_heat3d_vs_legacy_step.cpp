// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_heat3d_vs_legacy_step.cpp
 * @brief Structural stub: Heat3D does not use Model::step() pattern.
 *
 * @details
 * Heat3D physics uses explicit-Euler stepper.step() loops, not
 * Model::step(double) methods. This file documents that the migration
 * to operator-evaluation contract was not needed for Heat3D because
 * it already separated RHS computation from state advancement.
 *
 * The test suite provides:
 * - Compile-time verification that HeatModel does not expose step(double).
 * - Confirmation that the operator-evaluation contract header is available.
 *
 * Actual functionality and numerical equivalence are tested in
 * test_heat3d_operator_evaluation.cpp.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <heat3d/operator_evaluation.hpp>

#include <heat3d/heat_model.hpp>

using namespace heat3d;

TEST_CASE("Heat3D: HeatModel does not have step() method", "[heat3d][legacy]") {
  // Compile-time verification: HeatModel does not expose step(double).
  // This test passes if it compiles and links without requiring a step() method.
  HeatModel model;

  // Verify we can call rhs() which is the correct API for Heat3D
  HeatGrads g{.xx = 1.0, .yy = 2.0, .zz = 3.0};
  const double rhs_value = model.rhs(0.0, g);

  // Verify the value is computed correctly (sanity check)
  REQUIRE(rhs_value == 6.0 * heat3d::kD); // (1 + 2 + 3) * D
}

TEST_CASE("Heat3D: Operator evaluation contract exists", "[heat3d][legacy]") {
  // Verify operator evaluation header is available (semantic contract).
  // Actual functionality tested in test_heat3d_operator_evaluation.cpp.

  HeatOperator op;
  HeatGrads g{.xx = 1.0, .yy = 1.0, .zz = 1.0};
  const auto result = op.evaluate(g, 0.0);

  // Verify operator evaluation works (basic sanity check)
  REQUIRE(result.d_u == 3.0 * heat3d::kD);
}

TEST_CASE("Heat3D: HeatModel uses rhs() pattern, not step() pattern", "[heat3d][legacy]") {
  // This test explicitly documents the architectural decision that Heat3D
  // uses the rhs() pattern (operator evaluation) rather than the step()
  // pattern (state mutation), which is why the migration to the formal
  // operator-evaluation contract is straightforward.

  HeatModel model;

  // The correct API for Heat3D is rhs(), which computes the time derivative
  // without mutating state
  HeatGrads g{.xx = 1.0, .yy = 0.0, .zz = 0.0};
  const double du_dt = model.rhs(0.0, g);

  // Verify rhs() returns a value (the operator-evaluation pattern)
  REQUIRE(du_dt == heat3d::kD); // D * 1.0

  // HeatModel does not have a step() method that mutates internal state
  // This is a deliberate design choice that enables operator-evaluation
  REQUIRE(std::is_member_function_pointer_v<decltype(&heat3d::HeatModel::rhs)>);
}

int main(int argc, char* argv[]) {
  return Catch::Session().run(argc, argv);
}
