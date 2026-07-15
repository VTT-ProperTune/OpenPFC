// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json_world_time.hpp>
#include <openpfc/kernel/simulation/time.hpp>

using json = nlohmann::json;
using pfc::ui::from_json;
using pfc::Time;
using pfc::IntegratorMethod;

TEST_CASE("test_from_json_time_integrator_method") {
  SECTION("euler method") {
    json j = {
        {"timestepping", {{"t0", 0.0}, {"t1", 10.0}, {"dt", 0.1}, {"saveat", 1.0},
                          {"integrator", {{"method", "euler"}}}}}};
    auto time = from_json<Time>(j);
    REQUIRE(time.method() == IntegratorMethod::euler);
  }

  SECTION("rk2_heun method") {
    json j = {
        {"timestepping", {{"t0", 0.0}, {"t1", 10.0}, {"dt", 0.1}, {"saveat", 1.0},
                          {"integrator", {{"method", "rk2_heun"}}}}}};
    auto time = from_json<Time>(j);
    REQUIRE(time.method() == IntegratorMethod::rk2_heun);
  }

  SECTION("invalid method throws std::invalid_argument") {
    json j = {
        {"timestepping", {{"t0", 0.0}, {"t1", 10.0}, {"dt", 0.1}, {"saveat", 1.0},
                          {"integrator", {{"method", "unknown_method"}}}}}};
    REQUIRE_THROWS_AS(from_json<Time>(j), std::invalid_argument);
  }
}

TEST_CASE("test_from_json_time_missing_integrator_defaults_to_euler") {
  json j = {
      {"timestepping", {{"t0", 0.0}, {"t1", 10.0}, {"dt", 0.1}, {"saveat", 1.0}}}
  };
  auto time = from_json<Time>(j);
  REQUIRE(time.method() == IntegratorMethod::euler);
}
