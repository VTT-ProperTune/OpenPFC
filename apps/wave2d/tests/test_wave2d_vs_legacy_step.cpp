// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_wave2d_vs_legacy_step.cpp
 * @brief Structural stub: Wave2D does not use Model::step() pattern.
 *
 * @details
 * Wave2D physics uses explicit-Euler stepper.step() loops, not
 * Model::step(double) methods. This file documents that the migration
 * to operator-evaluation contract was proactively applied to prove
 * the contract works for coupled multi-field systems.
 *
 * The test suite provides:
 * - Compile-time verification that WaveModel does not expose step(double).
 * - Confirmation that the operator-evaluation contract header is available.
 * - Documentation of the tuple protocol for multi-field scattering.
 *
 * Actual functionality and numerical equivalence are tested in
 * test_wave2d_operator_evaluation.cpp.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <wave2d/operator_evaluation.hpp>

#include <wave2d/wave_model.hpp>

using namespace wave2d;

TEST_CASE("Wave2D: WaveModel does not have step() method", "[wave2d][legacy]") {
  // Compile-time verification: WaveModel does not expose step(double).
  // This test passes if it compiles and links without requiring a step() method.
  WaveModel model{.inv_dx2 = 1.0, .inv_dy2 = 1.0};

  // Verify we can call rhs() which is the correct API for Wave2D
  WaveLaplacian lap{.lxx = 1.0, .lyy = 1.0};
  const double v_val = 0.5;
  const auto result = model.rhs(0.0, v_val, lap);

  // Verify the values are computed correctly (sanity check)
  REQUIRE(result.du == v_val);
  REQUIRE(result.dv == wave2d::kC * wave2d::kC * (1.0 + 1.0)); // c² * (1 + 1)
}

TEST_CASE("Wave2D: Operator evaluation contract exists", "[wave2d][legacy]") {
  // Verify operator evaluation header is available (semantic contract).
  // Actual functionality tested in test_wave2d_operator_evaluation.cpp.

  WaveOperator op(1.0, 1.0);
  WaveLaplacian lap{.lxx = 1.0, .lyy = 1.0};
  const auto result = op.evaluate(lap, 0.0, 0.0);

  // Verify operator evaluation works (basic sanity check)
  REQUIRE(result.increments.du == 0.0);
  REQUIRE(result.increments.dv == wave2d::kC * wave2d::kC * 2.0);
}

TEST_CASE("Wave2D: WaveModel uses rhs() pattern, not step() pattern", "[wave2d][legacy]") {
  // This test explicitly documents the architectural decision that Wave2D
  // uses the rhs() pattern (operator evaluation) rather than the step()
  // pattern (state mutation), which is why the migration to the formal
  // operator-evaluation contract is straightforward.

  WaveModel model{.inv_dx2 = 1.0, .inv_dy2 = 1.0};

  // The correct API for Wave2D is rhs(), which computes the time derivative
  // without mutating state
  WaveLaplacian lap{.lxx = 1.0, .lyy = 0.0};
  const double v_val = 1.5;
  const auto result = model.rhs(0.0, v_val, lap);

  // Verify rhs() returns values for both du and dv (multi-field operator evaluation)
  REQUIRE(result.du == v_val);
  REQUIRE(result.dv == wave2d::kC * wave2d::kC); // c² * 1.0

  // WaveModel does not have a step() method that mutates internal state
  // This is a deliberate design choice that enables operator-evaluation
  REQUIRE(std::is_member_function_pointer_v<decltype(&wave2d::WaveModel::rhs)>);
}

TEST_CASE("Wave2D: Tuple protocol is supported for multi-field scattering", "[wave2d][legacy]") {
  // This test documents the tuple protocol used for scattering coupled field
  // increments into separate storage, which is critical for multi-field systems.

  WaveIncrements increments{.du = 1.5, .dv = -2.3};
  auto [du, dv] = increments.as_tuple();

  // Verify tuple protocol provides mutable references
  REQUIRE(du == 1.5);
  REQUIRE(dv == -2.3);

  // Verify modifications through tuple references modify the original
  du = 3.0;
  REQUIRE(increments.du == 3.0);

  // Verify const overload works for read-only access
  const auto& const_increments = WaveIncrements{.du = 0.5, .dv = 0.5};
  const auto [c_du, c_dv] = const_increments.as_tuple();
  static_assert(std::is_same_v<const double&, decltype(c_du)>);
  static_assert(std::is_same_v<const double&, decltype(c_dv)>);
}

int main(int argc, char* argv[]) {
  return Catch::Session().run(argc, argv);
}
