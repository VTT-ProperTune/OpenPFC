// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_integrator_method.cpp
 * @brief Unit tests for RKIntegratorMethod enum and validation utilities
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_tostring.hpp>
#include <openpfc/kernel/simulation/steppers/integrator_method.hpp>

using namespace pfc::sim::steppers;

// ============================================================================
// Enum coverage tests
// ============================================================================

TEST_CASE("test_enum_values_cover_all_methods") {
    // Verify all 5 enum values are defined and accessible
    REQUIRE(static_cast<int>(RKIntegratorMethod::Euler) == 0);
    REQUIRE(static_cast<int>(RKIntegratorMethod::RK2_Midpoint) == 1);
    REQUIRE(static_cast<int>(RKIntegratorMethod::RK2_Heun) == 2);
    REQUIRE(static_cast<int>(RKIntegratorMethod::RK4_Classical) == 3);
    REQUIRE(static_cast<int>(RKIntegratorMethod::BogackiShampine32) == 4);
}

// ============================================================================
// String conversion tests
// ============================================================================

TEST_CASE("test_to_string_returns_correct_names") {
    REQUIRE(to_string(RKIntegratorMethod::Euler) == "euler");
    REQUIRE(to_string(RKIntegratorMethod::RK2_Midpoint) == "rk2_midpoint");
    REQUIRE(to_string(RKIntegratorMethod::RK2_Heun) == "rk2_heun");
    REQUIRE(to_string(RKIntegratorMethod::RK4_Classical) == "rk4_classical");
    REQUIRE(to_string(RKIntegratorMethod::BogackiShampine32) == "bogacki_shampine32");
}

// ============================================================================
// Validation tests
// ============================================================================

TEST_CASE("test_validate_method_accepts_all_valid_methods") {
    // All methods should be valid without adaptive requirement
    REQUIRE_FALSE(validate_method(RKIntegratorMethod::Euler, false));
    REQUIRE_FALSE(validate_method(RKIntegratorMethod::RK2_Midpoint, false));
    REQUIRE_FALSE(validate_method(RKIntegratorMethod::RK2_Heun, false));
    REQUIRE_FALSE(validate_method(RKIntegratorMethod::RK4_Classical, false));
    REQUIRE_FALSE(validate_method(RKIntegratorMethod::BogackiShampine32, false));
}

TEST_CASE("test_validate_method_rejects_adaptive_without_estimator") {
    // Non-embedded methods should fail when requires_adaptive=true
    REQUIRE(validate_method(RKIntegratorMethod::Euler, true));
    REQUIRE(validate_method(RKIntegratorMethod::RK2_Midpoint, true));
    REQUIRE(validate_method(RKIntegratorMethod::RK2_Heun, true));
    REQUIRE(validate_method(RKIntegratorMethod::RK4_Classical, true));

    // Embedded method should pass
    REQUIRE_FALSE(validate_method(RKIntegratorMethod::BogackiShampine32, true));
}

TEST_CASE("test_validate_method_returns_descriptive_errors") {
    // Verify error messages contain method name and constraint description
    auto error = validate_method(RKIntegratorMethod::RK4_Classical, true);
    REQUIRE(error.has_value());
    REQUIRE(error->find("Adaptive step-size control") != std::string::npos);
    REQUIRE(error->find("embedded method") != std::string::npos);
    REQUIRE(error->find("rk4_classical") != std::string::npos);
}

// ============================================================================
// ButcherTableau factory tests
// ============================================================================

TEST_CASE("test_make_tableau_returns_correct_coefficients") {
    // Test that make_tableau returns appropriate tableau for each method
    auto euler_tableau = make_tableau(RKIntegratorMethod::Euler);
    REQUIRE(euler_tableau.stage_count() == 1);
    REQUIRE(std::string(euler_tableau.name()) == "Euler");

    auto rk2_midpoint_tableau = make_tableau(RKIntegratorMethod::RK2_Midpoint);
    REQUIRE(rk2_midpoint_tableau.stage_count() == 2);
    REQUIRE(std::string(rk2_midpoint_tableau.name()) == "RK2 midpoint");

    auto rk2_heun_tableau = make_tableau(RKIntegratorMethod::RK2_Heun);
    REQUIRE(rk2_heun_tableau.stage_count() == 2);
    REQUIRE(std::string(rk2_heun_tableau.name()) == "RK2 Heun");

    auto rk4_tableau = make_tableau(RKIntegratorMethod::RK4_Classical);
    REQUIRE(rk4_tableau.stage_count() == 4);
    REQUIRE(std::string(rk4_tableau.name()) == "RK4 classical");

    auto bs32_tableau = make_tableau(RKIntegratorMethod::BogackiShampine32);
    REQUIRE(bs32_tableau.stage_count() == 4);
    REQUIRE(std::string(bs32_tableau.name()) == "Bogacki-Shampine 3(2)");
}

TEST_CASE("test_make_euler_tableau_has_correct_coefficients") {
    // Test local Euler tableau coefficients: a_ij=[0], b_i=[1], c_i=[0]
    auto tableau = make_tableau(RKIntegratorMethod::Euler);

    REQUIRE(tableau.stage_count() == 1);

    // Check a_ij coefficients (1x1 matrix, should be [0])
    REQUIRE(tableau.a(0, 0) == 0.0);

    // Check b_i coefficients (should be [1])
    REQUIRE(tableau.b(0) == 1.0);

    // Check c_i coefficients (should be [0])
    REQUIRE(tableau.c(0) == 0.0);

    // Verify no embedded error estimator
    REQUIRE(tableau.has_embedded() == false);
}

// ============================================================================
// Embedded method detection tests
// ============================================================================

TEST_CASE("test_is_embedded_identifies_adaptive_methods") {
    // Only BogackiShampine32 should have embedded error estimator
    REQUIRE_FALSE(is_embedded(RKIntegratorMethod::Euler));
    REQUIRE_FALSE(is_embedded(RKIntegratorMethod::RK2_Midpoint));
    REQUIRE_FALSE(is_embedded(RKIntegratorMethod::RK2_Heun));
    REQUIRE_FALSE(is_embedded(RKIntegratorMethod::RK4_Classical));
    REQUIRE(is_embedded(RKIntegratorMethod::BogackiShampine32));
}
