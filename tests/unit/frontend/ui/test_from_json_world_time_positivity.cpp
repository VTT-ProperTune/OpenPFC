// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_from_json_world_time_positivity.cpp
 * @brief Catch2 coverage for positivity validation in from_json<World> and
 *        from_json<Time>
 */

#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <limits>
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

TEST_CASE("[world][positivity] from_json_world_rejects_zero_Lx") {
  json j = {
      {"Lx", 0},
      {"Ly", 128},
      {"Lz", 128},
      {"dx", 1.0},
      {"dy", 1.0},
      {"dz", 1.0},
      {"origin", "center"}};

  REQUIRE_THROWS_AS(from_json<World>(j), std::invalid_argument);
}

TEST_CASE("[world][positivity] from_json_world_rejects_negative_Lx") {
  json j = {
      {"Lx", -64},
      {"Ly", 128},
      {"Lz", 128},
      {"dx", 1.0},
      {"dy", 1.0},
      {"dz", 1.0},
      {"origin", "center"}};

  REQUIRE_THROWS_AS(from_json<World>(j), std::invalid_argument);
}

TEST_CASE("[world][positivity] from_json_world_rejects_zero_Ly") {
  json j = {
      {"Lx", 128},
      {"Ly", 0},
      {"Lz", 128},
      {"dx", 1.0},
      {"dy", 1.0},
      {"dz", 1.0},
      {"origin", "center"}};

  REQUIRE_THROWS_AS(from_json<World>(j), std::invalid_argument);
}

TEST_CASE("[world][positivity] from_json_world_rejects_negative_Ly") {
  json j = {
      {"Lx", 128},
      {"Ly", -32},
      {"Lz", 128},
      {"dx", 1.0},
      {"dy", 1.0},
      {"dz", 1.0},
      {"origin", "center"}};

  REQUIRE_THROWS_AS(from_json<World>(j), std::invalid_argument);
}

TEST_CASE("[world][positivity] from_json_world_rejects_zero_Lz") {
  json j = {
      {"Lx", 128},
      {"Ly", 128},
      {"Lz", 0},
      {"dx", 1.0},
      {"dy", 1.0},
      {"dz", 1.0},
      {"origin", "center"}};

  REQUIRE_THROWS_AS(from_json<World>(j), std::invalid_argument);
}

TEST_CASE("[world][positivity] from_json_world_rejects_negative_Lz") {
  json j = {
      {"Lx", 128},
      {"Ly", 128},
      {"Lz", -16},
      {"dx", 1.0},
      {"dy", 1.0},
      {"dz", 1.0},
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

TEST_CASE("[world][positivity] from_json_world_accepts_valid_positive_Lx_Ly_Lz") {
  json j = {
      {"Lx", 256},
      {"Ly", 64},
      {"Lz", 128},
      {"dx", 1.0},
      {"dy", 1.0},
      {"dz", 1.0},
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

TEST_CASE("[world][positivity] from_json_world_error_message_names_Lx") {
  json j = {
      {"Lx", 0},
      {"Ly", 128},
      {"Lz", 128},
      {"dx", 1.0},
      {"dy", 1.0},
      {"dz", 1.0},
      {"origin", "center"}};

  try {
    (void)from_json<World>(j);
    FAIL("Expected std::invalid_argument to be thrown");
  } catch (const std::invalid_argument &e) {
    std::string msg = e.what();
    REQUIRE(msg.find("Lx") != std::string::npos);
    REQUIRE(msg.find("number of grid points in X direction") != std::string::npos);
  }
}

TEST_CASE("[world][positivity] from_json_world_error_message_names_Ly") {
  json j = {
      {"Lx", 128},
      {"Ly", -32},
      {"Lz", 128},
      {"dx", 1.0},
      {"dy", 1.0},
      {"dz", 1.0},
      {"origin", "center"}};

  try {
    (void)from_json<World>(j);
    FAIL("Expected std::invalid_argument to be thrown");
  } catch (const std::invalid_argument &e) {
    std::string msg = e.what();
    REQUIRE(msg.find("Ly") != std::string::npos);
    REQUIRE(msg.find("number of grid points in Y direction") != std::string::npos);
  }
}

TEST_CASE("[world][positivity] from_json_world_error_message_names_Lz") {
  json j = {
      {"Lx", 128},
      {"Ly", 128},
      {"Lz", 0},
      {"dx", 1.0},
      {"dy", 1.0},
      {"dz", 1.0},
      {"origin", "center"}};

  try {
    (void)from_json<World>(j);
    FAIL("Expected std::invalid_argument to be thrown");
  } catch (const std::invalid_argument &e) {
    std::string msg = e.what();
    REQUIRE(msg.find("Lz") != std::string::npos);
    REQUIRE(msg.find("number of grid points in Z direction") != std::string::npos);
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

TEST_CASE("[world][positivity] from_json_world_with_nested_domain_rejects_zero_Lx") {
  json j = {
      {"domain", {{"Lx", 0}, {"Ly", 64}, {"Lz", 64}, {"dx", 1.0}, {"dy", 1.0}, {"dz", 1.0}}},
      {"origin", "center"}};

  REQUIRE_THROWS_AS(from_json<World>(j), std::invalid_argument);
}

TEST_CASE("[world][positivity] from_json_world_with_nested_domain_rejects_negative_Ly") {
  json j = {
      {"domain", {{"Lx", 64}, {"Ly", -32}, {"Lz", 64}, {"dx", 1.0}, {"dy", 1.0}, {"dz", 1.0}}},
      {"origin", "center"}};

  REQUIRE_THROWS_AS(from_json<World>(j), std::invalid_argument);
}

TEST_CASE("[world][positivity] from_json_world_with_nested_domain_rejects_zero_Lz") {
  json j = {
      {"domain", {{"Lx", 64}, {"Ly", 64}, {"Lz", 0}, {"dx", 1.0}, {"dy", 1.0}, {"dz", 1.0}}},
      {"origin", "center"}};

  REQUIRE_THROWS_AS(from_json<World>(j), std::invalid_argument);
}

TEST_CASE("[world][positivity] from_json_world_with_nested_domain_accepts_valid_positive_Lx_Ly_Lz") {
  json j = {
      {"domain", {{"Lx", 32}, {"Ly", 64}, {"Lz", 128}, {"dx", 1.0}, {"dy", 1.0}, {"dz", 1.0}}},
      {"origin", "center"}};

  REQUIRE_NOTHROW(from_json<World>(j));
}

TEST_CASE("[time][positivity] from_json_time_rejects_zero_saveat") {
  json j = {
      {"timestepping", {{"t0", 0.0}, {"t1", 10.0}, {"dt", 0.1}, {"saveat", 0.0}}}};

  REQUIRE_THROWS_AS(from_json<Time>(j), std::invalid_argument);
}

TEST_CASE("[time][positivity] from_json_time_rejects_negative_saveat") {
  json j = {
      {"timestepping", {{"t0", 0.0}, {"t1", 10.0}, {"dt", 0.1}, {"saveat", -1.0}}}};

  REQUIRE_THROWS_AS(from_json<Time>(j), std::invalid_argument);
}

TEST_CASE("[time][positivity] from_json_time_accepts_valid_positive_saveat") {
  json j = {
      {"timestepping", {{"t0", 0.0}, {"t1", 10.0}, {"dt", 0.1}, {"saveat", 0.5}}}};

  REQUIRE_NOTHROW(from_json<Time>(j));
}

TEST_CASE("[time][positivity] from_json_time_error_message_names_saveat") {
  json j = {
      {"timestepping", {{"t0", 0.0}, {"t1", 10.0}, {"dt", 0.1}, {"saveat", 0.0}}}};

  try {
    (void)from_json<Time>(j);
    FAIL("Expected std::invalid_argument to be thrown");
  } catch (const std::invalid_argument &e) {
    std::string msg = e.what();
    REQUIRE(msg.find("saveat") != std::string::npos);
    REQUIRE(msg.find("snapshot output interval") != std::string::npos);
  }
}

TEST_CASE("[time][positivity] from_json_time_rejects_t1_equal_to_t0") {
  json j = {
      {"timestepping", {{"t0", 5.0}, {"t1", 5.0}, {"dt", 0.1}, {"saveat", 1.0}}}};

  REQUIRE_THROWS_AS(from_json<Time>(j), std::invalid_argument);
}

TEST_CASE("[time][positivity] from_json_time_rejects_t1_less_than_t0") {
  json j = {
      {"timestepping", {{"t0", 10.0}, {"t1", 5.0}, {"dt", 0.1}, {"saveat", 1.0}}}};

  REQUIRE_THROWS_AS(from_json<Time>(j), std::invalid_argument);
}

TEST_CASE("[time][positivity] from_json_time_accepts_valid_t1_greater_than_t0") {
  json j = {
      {"timestepping", {{"t0", 0.0}, {"t1", 100.0}, {"dt", 0.1}, {"saveat", 1.0}}}};

  REQUIRE_NOTHROW(from_json<Time>(j));
}

TEST_CASE("[time][positivity] from_json_time_error_message_names_t1_interval") {
  json j = {
      {"timestepping", {{"t0", 10.0}, {"t1", 5.0}, {"dt", 0.1}, {"saveat", 1.0}}}};

  try {
    (void)from_json<Time>(j);
    FAIL("Expected std::invalid_argument to be thrown");
  } catch (const std::invalid_argument &e) {
    std::string msg = e.what();
    REQUIRE(msg.find("t1") != std::string::npos);
    REQUIRE(msg.find("simulation end time") != std::string::npos);
  }
}

TEST_CASE("[time][positivity] from_json_time_with_flat_structure_rejects_non_positive_saveat") {
  json j = {
      {"t0", 0.0},
      {"t1", 10.0},
      {"dt", 0.1},
      {"saveat", -0.5}};

  REQUIRE_THROWS_AS(from_json<Time>(j), std::invalid_argument);
}

TEST_CASE("[time][positivity] from_json_time_with_flat_structure_rejects_t1_less_than_t0") {
  json j = {
      {"t0", 8.0},
      {"t1", 2.0},
      {"dt", 0.1},
      {"saveat", 1.0}};

  REQUIRE_THROWS_AS(from_json<Time>(j), std::invalid_argument);
}

TEST_CASE("[world][positivity] from_json_world_rejects_nan_dx") {
  // Use quiet NaN which JSON can represent as a string that gets parsed
  // Construct JSON then override dx to NaN
  json j = {
      {"Lx", 128},
      {"Ly", 128},
      {"Lz", 128},
      {"dx", 1.0},
      {"dy", 1.0},
      {"dz", 1.0},
      {"origin", "center"}};

  // Set dx to NaN via direct manipulation of the JSON value
  std::vector<uint8_t> nan_bytes = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f}; // double NaN
  double nan_val;
  std::memcpy(&nan_val, nan_bytes.data(), sizeof(nan_val));
  j["dx"] = nan_val;

  REQUIRE_THROWS_AS(from_json<World>(j), std::invalid_argument);
}

TEST_CASE("[world][positivity] from_json_world_rejects_inf_dy") {
  // Construct JSON then override dy to Infinity
  json j = {
      {"Lx", 128},
      {"Ly", 128},
      {"Lz", 128},
      {"dx", 1.0},
      {"dy", 1.0},
      {"dz", 1.0},
      {"origin", "center"}};

  // Set dy to positive infinity
  j["dy"] = std::numeric_limits<double>::infinity();

  REQUIRE_THROWS_AS(from_json<World>(j), std::invalid_argument);
}

TEST_CASE("[world][positivity] from_json_world_rejects_negative_inf_dz") {
  // Construct JSON then override dz to -Infinity
  json j = {
      {"Lx", 128},
      {"Ly", 128},
      {"Lz", 128},
      {"dx", 1.0},
      {"dy", 1.0},
      {"dz", 1.0},
      {"origin", "center"}};

  // Set dz to negative infinity
  j["dz"] = -std::numeric_limits<double>::infinity();

  REQUIRE_THROWS_AS(from_json<World>(j), std::invalid_argument);
}

TEST_CASE("[time][positivity] from_json_time_rejects_nan_dt") {
  json j = {
      {"timestepping", {{"t0", 0.0}, {"t1", 10.0}, {"dt", 0.1}, {"saveat", 1.0}}}};

  // Set dt to NaN
  std::vector<uint8_t> nan_bytes = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f}; // double NaN
  double nan_val;
  std::memcpy(&nan_val, nan_bytes.data(), sizeof(nan_val));
  j["timestepping"]["dt"] = nan_val;

  REQUIRE_THROWS_AS(from_json<Time>(j), std::invalid_argument);
}

TEST_CASE("[time][positivity] from_json_time_rejects_inf_dt") {
  json j = {
      {"timestepping", {{"t0", 0.0}, {"t1", 10.0}, {"dt", 0.1}, {"saveat", 1.0}}}};

  // Set dt to positive infinity
  j["timestepping"]["dt"] = std::numeric_limits<double>::infinity();

  REQUIRE_THROWS_AS(from_json<Time>(j), std::invalid_argument);
}

TEST_CASE("[time][positivity] from_json_time_rejects_inf_saveat") {
  json j = {
      {"timestepping", {{"t0", 0.0}, {"t1", 10.0}, {"dt", 0.1}, {"saveat", 1.0}}}};

  // Set saveat to positive infinity
  j["timestepping"]["saveat"] = std::numeric_limits<double>::infinity();

  REQUIRE_THROWS_AS(from_json<Time>(j), std::invalid_argument);
}

TEST_CASE("[time][positivity] from_json_time_rejects_nan_t0") {
  json j = {
      {"timestepping", {{"t0", 0.0}, {"t1", 10.0}, {"dt", 0.1}, {"saveat", 1.0}}}};

  // Set t0 to NaN
  std::vector<uint8_t> nan_bytes = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f}; // double NaN
  double nan_val;
  std::memcpy(&nan_val, nan_bytes.data(), sizeof(nan_val));
  j["timestepping"]["t0"] = nan_val;

  REQUIRE_THROWS_AS(from_json<Time>(j), std::invalid_argument);
}

TEST_CASE("[time][positivity] from_json_time_rejects_inf_t1") {
  json j = {
      {"timestepping", {{"t0", 0.0}, {"t1", 10.0}, {"dt", 0.1}, {"saveat", 1.0}}}};

  // Set t1 to positive infinity
  j["timestepping"]["t1"] = std::numeric_limits<double>::infinity();

  REQUIRE_THROWS_AS(from_json<Time>(j), std::invalid_argument);
}

TEST_CASE("[world][positivity] from_json_world_with_nested_domain_rejects_nan_dx") {
  json j = {
      {"domain", {{"Lx", 64}, {"Ly", 64}, {"Lz", 64}, {"dx", 1.0}, {"dy", 1.0}, {"dz", 1.0}}},
      {"origin", "center"}};

  // Set dx to NaN
  std::vector<uint8_t> nan_bytes = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f}; // double NaN
  double nan_val;
  std::memcpy(&nan_val, nan_bytes.data(), sizeof(nan_val));
  j["domain"]["dx"] = nan_val;

  REQUIRE_THROWS_AS(from_json<World>(j), std::invalid_argument);
}
