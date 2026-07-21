// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_from_json_integrator_method.cpp
 * @brief Catch2 coverage for from_json<RKIntegratorMethod>
 */

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json_integrator_method.hpp>
#include <openpfc/kernel/simulation/steppers/integrator_method.hpp>

using namespace pfc::sim::steppers;

TEST_CASE("test_from_json_deserializes_valid_methods") {
  using pfc::ui::from_json;

  REQUIRE(from_json<RKIntegratorMethod>(nlohmann::json("euler")) ==
          RKIntegratorMethod::Euler);
  REQUIRE(from_json<RKIntegratorMethod>(nlohmann::json("rk2_midpoint")) ==
          RKIntegratorMethod::RK2_Midpoint);
  REQUIRE(from_json<RKIntegratorMethod>(nlohmann::json("rk2_heun")) ==
          RKIntegratorMethod::RK2_Heun);
  REQUIRE(from_json<RKIntegratorMethod>(nlohmann::json("rk4_classical")) ==
          RKIntegratorMethod::RK4_Classical);
  REQUIRE(from_json<RKIntegratorMethod>(nlohmann::json("bogacki_shampine32")) ==
          RKIntegratorMethod::BogackiShampine32);
}

TEST_CASE("test_from_json_throws_on_unknown_string") {
  using pfc::ui::from_json;

  REQUIRE_THROWS_AS(from_json<RKIntegratorMethod>(nlohmann::json("unknown_method")),
                    std::runtime_error);
  REQUIRE_THROWS_AS(from_json<RKIntegratorMethod>(nlohmann::json("RK4")),
                    std::runtime_error);
  REQUIRE_THROWS_AS(from_json<RKIntegratorMethod>(nlohmann::json("euler ")),
                    std::runtime_error);

  // Verify error message is descriptive
  try {
    (void)from_json<RKIntegratorMethod>(nlohmann::json("invalid"));
    FAIL("Expected std::runtime_error to be thrown");
  } catch (const std::runtime_error &e) {
    std::string msg = e.what();
    REQUIRE(msg.find("Unknown RK integrator method") != std::string::npos);
    REQUIRE(msg.find("invalid") != std::string::npos);
    REQUIRE(msg.find("Valid methods are") != std::string::npos);
  }
}
