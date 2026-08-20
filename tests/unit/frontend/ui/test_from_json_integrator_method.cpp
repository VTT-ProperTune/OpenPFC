// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_from_json_integrator_method.cpp
 * @brief Catch2 coverage for from_json<RKIntegratorMethod>
 */

#include <stdexcept>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json_integrator_method.hpp>
#include <openpfc/kernel/simulation/steppers/integrator_method.hpp>

using Catch::Matchers::ContainsSubstring;

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
  REQUIRE(from_json<RKIntegratorMethod>(nlohmann::json("imex_euler")) ==
          RKIntegratorMethod::ImexEuler);
  REQUIRE(from_json<RKIntegratorMethod>(nlohmann::json("etd1")) ==
          RKIntegratorMethod::ETD1);
}

TEST_CASE("test_from_json_throws_on_unknown_string") {
  using pfc::ui::from_json;

  REQUIRE_THROWS_AS(from_json<RKIntegratorMethod>(nlohmann::json("unknown_method")),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(from_json<RKIntegratorMethod>(nlohmann::json("RK4")),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(from_json<RKIntegratorMethod>(nlohmann::json("euler ")),
                    std::invalid_argument);

  try {
    (void)from_json<RKIntegratorMethod>(nlohmann::json("invalid"));
    FAIL("Expected std::invalid_argument to be thrown");
  } catch (const std::invalid_argument &e) {
    REQUIRE_THAT(std::string(e.what()), ContainsSubstring("method"));
    REQUIRE_THAT(std::string(e.what()), ContainsSubstring("invalid"));
    REQUIRE_THAT(std::string(e.what()), ContainsSubstring("euler"));
  }
}
