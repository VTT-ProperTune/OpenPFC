// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_from_json_world_time_positivity.cpp
 * @brief Catch2 coverage for positivity validation in from_json<World> and
 *        from_json<Time>
 */

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json_world_time.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/simulation/time.hpp>

using json = nlohmann::json;

using pfc::ui::from_json;
using pfc::World;
using pfc::Time;

TEST_CASE("[world][positivity] from_json_world_rejects_zero_dx") {
  json j = {
      {"Lx", 128},
      {"Ly", 128},
      {"Lz", 128},
      {"dx", 0.0},
      {"dy", 1.0},
      {"dz", 1.0},
      {"origin", "center"}};

  REQUIRE_THROWS_AS(from_json<World>(j), std::invalid_argument);
}

TEST_CASE("[world][positivity] from_json_world_rejects_negative_dx") {
  json j = {
      {"Lx", 128},
      {"Ly", 128},
      {"Lz", 128},
      {"dx", -1.0},
      {"dy", 1.0},
      {"dz", 1.0},
      {"origin", "center"}};

  REQUIRE_THROWS_AS(from_json<World>(j), std::invalid_argument);
}

TEST_CASE("[world][positivity] from_json_world_rejects_zero_dy") {
  json j = {
      {"Lx", 128},
      {"Ly", 128},
      {"Lz", 128},
      {"dx", 1.0},
      {"dy", 0.0},
      {"dz", 1.0},
      {"origin", "center"}};

  REQUIRE_THROWS_AS(from_json<World>(j), std::invalid_argument);
}

TEST_CASE("[world][positivity] from_json_world_rejects_negative_dy") {
  json j = {
      {"Lx", 128},
      {"Ly", 128},
      {"Lz", 128},
      {"dx", 1.0},
      {"dy", -0.5},
      {"dz", 1.0},
      {"origin", "center"}};

  REQUIRE_THROWS_AS(from_json<World>(j), std::invalid_argument);
}

TEST_CASE("[world][positivity] from_json_world_rejects_zero_dz") {
  json j = {
      {"Lx", 128},
      {"Ly", 128},
      {"Lz", 128},
      {"dx", 1.0},
      {"dy", 1.0},
      {"dz", 0.0},
      {"origin", "center"}};

  REQUIRE_THROWS_AS(from_json<World>(j), std::invalid_argument);
}

TEST_CASE("[world][positivity] from_json_world_rejects_negative_dz") {
  json j = {
      {"Lx", 128},
      {"Ly", 128},
      {"Lz", 128},
      {"dx", 1.0},
      {"dy", 1.0},
      {"dz", -2.0},
      {"origin", "center"}};

  REQUIRE_THROWS_AS(from_json<World>(j), std::invalid_argument);
}

TEST_CASE("[world][positivity] from_json_world_accepts_valid_positive_dx_dy_dz") {
  json j = {
      {"Lx", 128},
      {"Ly", 128},
      {"Lz", 128},
      {"dx", 1.5},
      {"dy", 0.01},
      {"dz", 4.7},
      {"origin", "center"}};

  REQUIRE_NOTHROW(from_json<World>(j));
}

TEST_CASE("[world][positivity] from_json_world_error_message_names_dx") {
  json j = {
      {"Lx", 128},
      {"Ly", 128},
      {"Lz", 128},
      {"dx", 0.0},
      {"dy", 1.0},
      {"dz", 1.0},
      {"origin", "center"}};

  try {
    (void)from_json<World>(j);
    FAIL("Expected std::invalid_argument to be thrown");
  } catch (const std::invalid_argument &e) {
    std::string msg = e.what();
    REQUIRE(msg.find("dx") != std::string::npos);
    REQUIRE(msg.find("grid spacing in X direction") != std::string::npos);
  }
}

TEST_CASE("[world][positivity] from_json_world_error_message_names_dy") {
  json j = {
      {"Lx", 128},
      {"Ly", 128},
      {"Lz", 128},
      {"dx", 1.0},
      {"dy", -1.0},
      {"dz", 1.0},
      {"origin", "center"}};

  try {
    (void)from_json<World>(j);
    FAIL("Expected std::invalid_argument to be thrown");
  } catch (const std::invalid_argument &e) {
    std::string msg = e.what();
    REQUIRE(msg.find("dy") != std::string::npos);
    REQUIRE(msg.find("grid spacing in Y direction") != std::string::npos);
  }
}

TEST_CASE("[world][positivity] from_json_world_error_message_names_dz") {
  json j = {
      {"Lx", 128},
      {"Ly", 128},
      {"Lz", 128},
      {"dx", 1.0},
      {"dy", 1.0},
      {"dz", -0.001},
      {"origin", "center"}};

  try {
    (void)from_json<World>(j);
    FAIL("Expected std::invalid_argument to be thrown");
  } catch (const std::invalid_argument &e) {
    std::string msg = e.what();
    REQUIRE(msg.find("dz") != std::string::npos);
    REQUIRE(msg.find("grid spacing in Z direction") != std::string::npos);
  }
}

TEST_CASE("[time][positivity] from_json_time_rejects_zero_dt") {
  json j = {
      {"timestepping", {{"t0", 0.0}, {"t1", 10.0}, {"dt", 0.0}, {"saveat", 1.0}}}};

  REQUIRE_THROWS_AS(from_json<Time>(j), std::invalid_argument);
}

TEST_CASE("[time][positivity] from_json_time_rejects_negative_dt") {
  json j = {
      {"timestepping", {{"t0", 0.0}, {"t1", 10.0}, {"dt", -0.01}, {"saveat", 1.0}}}};

  REQUIRE_THROWS_AS(from_json<Time>(j), std::invalid_argument);
}

TEST_CASE("[time][positivity] from_json_time_accepts_valid_positive_dt") {
  json j = {
      {"timestepping", {{"t0", 0.0}, {"t1", 10.0}, {"dt", 0.1}, {"saveat", 1.0}}}};

  REQUIRE_NOTHROW(from_json<Time>(j));
}

TEST_CASE("[time][positivity] from_json_time_accepts_very_small_positive_dt") {
  json j = {
      {"timestepping", {{"t0", 0.0}, {"t1", 10.0}, {"dt", 1e-8}, {"saveat", 1.0}}}};

  REQUIRE_NOTHROW(from_json<Time>(j));
}

TEST_CASE("[time][positivity] from_json_time_error_message_names_dt") {
  json j = {
      {"timestepping", {{"t0", 0.0}, {"t1", 10.0}, {"dt", 0.0}, {"saveat", 1.0}}}};

  try {
    (void)from_json<Time>(j);
    FAIL("Expected std::invalid_argument to be thrown");
  } catch (const std::invalid_argument &e) {
    std::string msg = e.what();
    REQUIRE(msg.find("dt") != std::string::npos);
    REQUIRE(msg.find("timestep size") != std::string::npos);
  }
}

TEST_CASE("[world][positivity] from_json_world_with_nested_domain") {
  json j = {
      {"domain", {{"Lx", 64}, {"Ly", 64}, {"Lz", 64}, {"dx", 0.0}, {"dy", 1.0}, {"dz", 1.0}}},
      {"origin", "center"}};

  REQUIRE_THROWS_AS(from_json<World>(j), std::invalid_argument);

  // Test with positive value should work
  j["domain"]["dx"] = 1.5;
  REQUIRE_NOTHROW(from_json<World>(j));
}
