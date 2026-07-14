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
  REQUIRE(tableau.order() == 2);
  REQUIRE(tableau.name() == "Bogacki-Shampine 2(3)");
  REQUIRE(tableau.a(1, 0) == Catch::Approx(0.5));
  REQUIRE(tableau.a(2, 1) == Catch::Approx(0.75));
  REQUIRE(tableau.a(3, 0) == Catch::Approx(2.0/9.0));
  REQUIRE(tableau.a(3, 1) == Catch::Approx(1.0/3.0));
  REQUIRE(tableau.a(3, 2) == Catch::Approx(4.0/9.0));
  REQUIRE(tableau.b_hat(0) == Catch::Approx(2.0/9.0));
  REQUIRE(tableau.b_hat(1) == Catch::Approx(1.0/3.0));
  REQUIRE(tableau.b_hat(2) == Catch::Approx(4.0/9.0));
  REQUIRE(tableau.b_hat(3) == 0.0);
}

TEST_CASE("Negative coefficients are accepted") {
  auto tableau = pfc::sim::steppers::make_embedded_rk23<double>();
  
  pfc::sim::steppers::ButcherTableau<double> tableau_neg(
    2,
    {0.0, 0.0, -0.5, 0.0},
    {0.5, 0.5},
    {0.0, -0.5},
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
    {0.0},
    {1.0},
    {0.0},
    {},
    "",
    0
  );
  REQUIRE(tableau.name() == "");
  REQUIRE(tableau.order() == 0);
  REQUIRE(tableau.stage_count() == 1);
}

TEST_CASE("Coefficients are const after construction") {
  auto tableau = pfc::sim::steppers::make_rk4_classical<double>();
  REQUIRE(tableau.a(0, 0) != 999.0);
  REQUIRE_NOTHROW(tableau.a(0, 0));
  REQUIRE(tableau.b(0) == tableau.b(0));
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
    {0.0, 0.0, 0.0, 0.0},
    {1.0},
    {0.0, 1.0},
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
  REQUIRE_THROWS_AS(pfc::sim::steppers::ButcherTableau<double>(
    2,
    {0.0, 0.0, 0.5, 0.5},
    {0.5, 0.5},
    {0.5, 1.0},
    {},
    "",
    0
  ), pfc::sim::steppers::TableauValidationError);
  
  REQUIRE_THROWS_AS(pfc::sim::steppers::ButcherTableau<double>(
    2,
    {0.0, 0.5, 0.0, 0.0},
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
  REQUIRE_THROWS_AS(pfc::sim::steppers::ButcherTableau<double>(
    2,
    {0.0, 0.0, 0.5, 0.0},
    {0.0, 1.0},
    {0.0, 0.3},
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
    REQUIRE(msg.find("0.5") != std::string::npos);
    REQUIRE(msg.find("0.3") != std::string::npos);
  }
}

TEST_CASE("NaN coefficient is rejected") {
  double nan = std::numeric_limits<double>::quiet_NaN();
  REQUIRE_THROWS_AS(pfc::sim::steppers::ButcherTableau<double>(
    1, {nan}, {1.0}, {0.0}, {}, "", 0
  ), pfc::sim::steppers::TableauValidationError);
  
  REQUIRE_THROWS_AS(pfc::sim::steppers::ButcherTableau<double>(
    1, {0.0}, {nan}, {0.0}, {}, "", 0
  ), pfc::sim::steppers::TableauValidationError);
  
  REQUIRE_THROWS_AS(pfc::sim::steppers::ButcherTableau<double>(
    1, {0.0}, {1.0}, {nan}, {}, "", 0
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
    1, {inf}, {1.0}, {0.0}, {}, "", 0
  ), pfc::sim::steppers::TableauValidationError);
  
  REQUIRE_THROWS_AS(pfc::sim::steppers::ButcherTableau<double>(
    1, {0.0}, {-inf}, {0.0}, {}, "", 0
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
