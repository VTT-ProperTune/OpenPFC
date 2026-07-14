// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <openpfc/kernel/simulation/steppers/butcher_tableau.hpp>
#include <limits>

TEST_CASE("RK2 midpoint tableau is valid") {
  auto tableau = pfc::sim::steppers::make_rk2_midpoint<double>();
  REQUIRE(tableau.stage_count() == 2);
  REQUIRE(tableau.a(0, 0) == 0.0);
  REQUIRE(tableau.a(1, 0) == 0.5);
  REQUIRE(tableau.a(1, 1) == 0.0);
  REQUIRE(tableau.b(0) == 0.0);
  REQUIRE(tableau.b(1) == 1.0);
  REQUIRE(tableau.c(0) == 0.0);
  REQUIRE(tableau.c(1) == 0.5);
  REQUIRE(tableau.name() == "RK2 midpoint");
  REQUIRE(tableau.order() == 2);
  REQUIRE(!tableau.has_embedded());
}

TEST_CASE("RK2 Heun tableau is valid") {
  auto tableau = pfc::sim::steppers::make_rk2_heun<double>();
  REQUIRE(tableau.stage_count() == 2);
  REQUIRE(tableau.name() == "RK2 Heun");
  REQUIRE(tableau.order() == 2);
  REQUIRE(tableau.c(0) == 0.0);
  REQUIRE(tableau.c(1) == 1.0);
  REQUIRE(tableau.b(0) == Catch::Approx(0.5));
  REQUIRE(tableau.b(1) == Catch::Approx(0.5));
}

TEST_CASE("RK4 classical tableau is valid") {
  auto tableau = pfc::sim::steppers::make_rk4_classical<double>();
  REQUIRE(tableau.stage_count() == 4);
  REQUIRE(tableau.b(0) == Catch::Approx(1.0/6.0));
  REQUIRE(tableau.b(1) == Catch::Approx(1.0/3.0));
  REQUIRE(tableau.b(2) == Catch::Approx(1.0/3.0));
  REQUIRE(tableau.b(3) == Catch::Approx(1.0/6.0));
  REQUIRE(tableau.c(0) == 0.0);
  REQUIRE(tableau.c(1) == 0.5);
  REQUIRE(tableau.c(2) == 0.5);
  REQUIRE(tableau.c(3) == 1.0);
  REQUIRE(tableau.name() == "RK4 classical");
  REQUIRE(tableau.order() == 4);
}

TEST_CASE("Embedded RK23 (Bogacki-Shampine) tableau is valid") {
  auto tableau = pfc::sim::steppers::make_embedded_rk23<double>();
  REQUIRE(tableau.stage_count() == 4);
  REQUIRE(tableau.has_embedded());
  REQUIRE(tableau.order() == 3);
  REQUIRE(tableau.embedded_order() == 2);
  REQUIRE(tableau.name() == "Bogacki-Shampine 3(2)");
  REQUIRE(tableau.a(1, 0) == Catch::Approx(0.5));
  REQUIRE(tableau.a(2, 1) == Catch::Approx(0.75));
  REQUIRE(tableau.a(3, 0) == Catch::Approx(2.0/9.0));
  REQUIRE(tableau.a(3, 1) == Catch::Approx(1.0/3.0));
  REQUIRE(tableau.a(3, 2) == Catch::Approx(4.0/9.0));

  // Primary third-order output weights
  REQUIRE(tableau.b(0) == Catch::Approx(2.0/9.0));
  REQUIRE(tableau.b(1) == Catch::Approx(1.0/3.0));
  REQUIRE(tableau.b(2) == Catch::Approx(4.0/9.0));
  REQUIRE(tableau.b(3) == 0.0);

  // Embedded second-order error estimator
  REQUIRE(tableau.b_hat(0) == Catch::Approx(7.0/24.0));
  REQUIRE(tableau.b_hat(1) == Catch::Approx(1.0/4.0));
  REQUIRE(tableau.b_hat(2) == Catch::Approx(1.0/3.0));
  REQUIRE(tableau.b_hat(3) == Catch::Approx(1.0/8.0));
}

TEST_CASE("Float type coefficients work correctly") {
  auto tableau = pfc::sim::steppers::make_rk2_midpoint<float>();
  REQUIRE(tableau.stage_count() == 2);
  REQUIRE(tableau.a(1, 0) == Catch::Approx(0.5f));
  REQUIRE(tableau.b(1) == 1.0f);
  REQUIRE(tableau.c(1) == 0.5f);
  REQUIRE(tableau.name() == "RK2 midpoint");
  REQUIRE(tableau.order() == 2);
}

TEST_CASE("Float type RK4 classical tableau is valid") {
  auto tableau = pfc::sim::steppers::make_rk4_classical<float>();
  REQUIRE(tableau.stage_count() == 4);
  REQUIRE(tableau.b(0) == Catch::Approx(1.0f/6.0f));
  REQUIRE(tableau.b(1) == Catch::Approx(1.0f/3.0f));
  REQUIRE(tableau.b(2) == Catch::Approx(1.0f/3.0f));
  REQUIRE(tableau.b(3) == Catch::Approx(1.0f/6.0f));
  REQUIRE(tableau.c(1) == 0.5f);
  REQUIRE(tableau.c(2) == 0.5f);
  REQUIRE(tableau.c(3) == 1.0f);
  REQUIRE(tableau.order() == 4);
}

TEST_CASE("Negative coefficients are accepted") {
  // Construct tableau with explicit negative coefficients
  pfc::sim::steppers::ButcherTableau<double> tableau_neg(
    2,
    {0.0, 0.0, -0.5, 0.0},  // a_ij with negative entry
    {0.5, 0.5},              // b_i
    {0.0, -0.5},             // c_i negative but row sum matches
    {},
    "",
    0
  );
  REQUIRE(tableau_neg.stage_count() == 2);
  REQUIRE(tableau_neg.a(1, 0) == -0.5);
  REQUIRE(tableau_neg.c(1) == -0.5);
}

TEST_CASE("Accessors return correct coefficient values") {
  auto tableau = pfc::sim::steppers::make_rk2_midpoint<double>();
  REQUIRE(tableau.a(1, 0) == 0.5);
  REQUIRE(tableau.b(1) == 1.0);
  REQUIRE(tableau.c(1) == 0.5);
}

TEST_CASE("b_hat access throws when not present") {
  auto tableau = pfc::sim::steppers::make_rk2_midpoint<double>();
  REQUIRE_THROWS_AS(tableau.b_hat(0), std::runtime_error);
}

TEST_CASE("Metadata fields are optional") {
  pfc::sim::steppers::ButcherTableau<double> tableau(
    1,
    {0.0},  // a_ij
    {1.0},  // b_i
    {0.0},  // c_i
    {},     // no b_hat
    "",     // empty name
    0       // zero order
  );
  REQUIRE(tableau.name() == "");
  REQUIRE(tableau.order() == 0);
  REQUIRE(tableau.stage_count() == 1);
}

TEST_CASE("Coefficients are immutable via API") {
  // Accessors return values, not references, ensuring immutability
  auto tableau = pfc::sim::steppers::make_rk4_classical<double>();

  // Modifications to returned values do not affect tableau
  const double original_b0 = tableau.b(0);
  double copy_b0 = original_b0;
  copy_b0 = 999.0;
  REQUIRE(tableau.b(0) == original_b0);
  REQUIRE(tableau.b(0) != 999.0);

  // No mutable reference access is provided (verified by API design)
  // All coefficient accessors return by value, not by reference
  static_assert(std::is_same_v<decltype(tableau.b(0)), double>);

  // Multiple accesses return consistent values
  REQUIRE(tableau.b(0) == tableau.b(0));
  REQUIRE(tableau.a(1, 0) == tableau.a(1, 0));
}

TEST_CASE("Zero stage count is rejected") {
  REQUIRE_THROWS_AS(pfc::sim::steppers::ButcherTableau<double>(
    0, {}, {}, {}, {}, "", 0
  ), pfc::sim::steppers::TableauValidationError);

  try {
    pfc::sim::steppers::ButcherTableau<double>(
      0, {}, {}, {}, {}, "", 0);
  } catch (const pfc::sim::steppers::TableauValidationError& err) {
    REQUIRE(err.error_type() == pfc::sim::steppers::TableauValidationError::ErrorType::InvalidStageCount);
    std::string msg = err.what();
    REQUIRE(msg.find(">= 1") != std::string::npos);
  }
}

TEST_CASE("Stage count mismatch is rejected") {
  REQUIRE_THROWS_AS(pfc::sim::steppers::ButcherTableau<double>(
    2,
    {0.0, 0.0, 0.0, 0.0},  // a_ij OK (2x2)
    {1.0},                  // b_i: only 1 element (should be 2)
    {0.0, 1.0},             // c_i OK (2 elements)
    {},
    "",
    0
  ), pfc::sim::steppers::TableauValidationError);

  try {
    pfc::sim::steppers::ButcherTableau<double>(
      2, {0.0, 0.0, 0.0, 0.0}, {1.0}, {0.0, 1.0}, {}, "", 0);
  } catch (const pfc::sim::steppers::TableauValidationError& err) {
    REQUIRE(err.error_type() == pfc::sim::steppers::TableauValidationError::ErrorType::StageCountMismatch);
    std::string msg = err.what();
    REQUIRE(msg.find("b_i") != std::string::npos);
  }
}

TEST_CASE("Explicit structure violation is rejected") {
  // Non-zero on diagonal: a[1][1] = 0.5 violates explicit structure (must be zero for i <= j)
  REQUIRE_THROWS_AS(pfc::sim::steppers::ButcherTableau<double>(
    2,
    {0.0, 0.0, 0.5, 0.5},  // a_ij: a[1][1]=0.5 NON-ZERO (should be 0 for i <= j)
    {0.5, 0.5},
    {0.5, 1.0},
    {},
    "",
    0
  ), pfc::sim::steppers::TableauValidationError);

  // Non-zero above diagonal: a[0][1] = 0.5 violates explicit structure
  REQUIRE_THROWS_AS(pfc::sim::steppers::ButcherTableau<double>(
    2,
    {0.0, 0.5, 0.0, 0.0},  // a_ij: a[0][1]=0.5 NON-ZERO for i < j
    {0.5, 0.5},
    {0.5, 1.0},
    {},
    "",
    0
  ), pfc::sim::steppers::TableauValidationError);

  try {
    pfc::sim::steppers::ButcherTableau<double>(
      2, {0.0, 0.5, 0.0, 0.0}, {0.5, 0.5}, {0.5, 1.0}, {}, "", 0);
  } catch (const pfc::sim::steppers::TableauValidationError& err) {
    REQUIRE(err.error_type() == pfc::sim::steppers::TableauValidationError::ErrorType::ExplicitStructureViolation);
    std::string msg = err.what();
    REQUIRE(msg.find("0") != std::string::npos);
    REQUIRE(msg.find("a_ij") != std::string::npos);
    REQUIRE(msg.find("i <= j") != std::string::npos);
  }
}

TEST_CASE("Row sum inconsistency is rejected") {
  // a_ij row sum doesn't match c_i
  REQUIRE_THROWS_AS(pfc::sim::steppers::ButcherTableau<double>(
    2,
    {0.0, 0.0, 0.5, 0.0},  // row 0 sum = 0, row 1 sum = 0.5
    {0.0, 1.0},
    {0.0, 0.3},  // c[1] = 0.3 != 0.5 (row 1 sum)
    {},
    "",
    0
  ), pfc::sim::steppers::TableauValidationError);

  try {
    pfc::sim::steppers::ButcherTableau<double>(
      2, {0.0, 0.0, 0.5, 0.0}, {0.0, 1.0}, {0.0, 0.3}, {}, "", 0);
  } catch (const pfc::sim::steppers::TableauValidationError& err) {
    REQUIRE(err.error_type() == pfc::sim::steppers::TableauValidationError::ErrorType::RowSumInconsistency);
    std::string msg = err.what();
    REQUIRE(msg.find("0.5") != std::string::npos);  // actual row sum
    REQUIRE(msg.find("0.3") != std::string::npos);  // expected c_i value
  }
}

TEST_CASE("NaN coefficient is rejected") {
  double nan = std::numeric_limits<double>::quiet_NaN();
  REQUIRE_THROWS_AS(pfc::sim::steppers::ButcherTableau<double>(
    1,
    {nan},  // a_ij with NaN
    {1.0},
    {0.0},
    {},
    "",
    0
  ), pfc::sim::steppers::TableauValidationError);

  REQUIRE_THROWS_AS(pfc::sim::steppers::ButcherTableau<double>(
    1, {0.0}, {nan}, {0.0}, {}, "", 0  // b_i with NaN
  ), pfc::sim::steppers::TableauValidationError);

  REQUIRE_THROWS_AS(pfc::sim::steppers::ButcherTableau<double>(
    1, {0.0}, {1.0}, {nan}, {}, "", 0  // c_i with NaN
  ), pfc::sim::steppers::TableauValidationError);

  try {
    pfc::sim::steppers::ButcherTableau<double>(
      1, {nan}, {1.0}, {0.0}, {}, "", 0);
  } catch (const pfc::sim::steppers::TableauValidationError& err) {
    REQUIRE(err.error_type() == pfc::sim::steppers::TableauValidationError::ErrorType::NonFiniteCoefficient);
    std::string msg = err.what();
    REQUIRE(msg.find("a_ij") != std::string::npos);
    REQUIRE(msg.find("not finite") != std::string::npos);
  }
}

TEST_CASE("Inf coefficient is rejected") {
  double inf = std::numeric_limits<double>::infinity();
  REQUIRE_THROWS_AS(pfc::sim::steppers::ButcherTableau<double>(
    1,
    {inf},  // a_ij with Inf
    {1.0},
    {0.0},
    {},
    "",
    0
  ), pfc::sim::steppers::TableauValidationError);

  REQUIRE_THROWS_AS(pfc::sim::steppers::ButcherTableau<double>(
    1, {0.0}, {-inf}, {0.0}, {}, "", 0  // b_i with -Inf
  ), pfc::sim::steppers::TableauValidationError);

  try {
    pfc::sim::steppers::ButcherTableau<double>(
      1, {inf}, {1.0}, {0.0}, {}, "", 0);
  } catch (const pfc::sim::steppers::TableauValidationError& err) {
    REQUIRE(err.error_type() == pfc::sim::steppers::TableauValidationError::ErrorType::NonFiniteCoefficient);
    std::string msg = err.what();
    REQUIRE(msg.find("a_ij") != std::string::npos);
    REQUIRE(msg.find("not finite") != std::string::npos);
  }
}
