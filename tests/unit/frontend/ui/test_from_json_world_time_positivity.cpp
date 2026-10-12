// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_from_json_world_time_positivity.cpp
 * @brief Catch2 coverage for positivity validation in from_json<Time>.
 *        The domain parser has its own file, test_from_json_domain.cpp.
 */

#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json_world_time.hpp>
#include <openpfc/kernel/simulation/time.hpp>

using json = nlohmann::json;

using pfc::Time;
using pfc::ui::from_json;

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

TEST_CASE("[time][positivity] from_json_time_accepts_zero_saveat_to_disable_saves") {
  json j = {
      {"timestepping", {{"t0", 0.0}, {"t1", 10.0}, {"dt", 0.1}, {"saveat", 0.0}}}};

  const Time time = from_json<Time>(j);
  REQUIRE(time.get_saveat() == 0.0);
}

TEST_CASE(
    "[time][positivity] from_json_time_accepts_negative_saveat_to_disable_saves") {
  json j = {
      {"timestepping", {{"t0", 0.0}, {"t1", 10.0}, {"dt", 0.1}, {"saveat", -1.0}}}};

  const Time time = from_json<Time>(j);
  REQUIRE(time.get_saveat() == -1.0);
}

TEST_CASE("[time][positivity] from_json_time_accepts_valid_positive_saveat") {
  json j = {
      {"timestepping", {{"t0", 0.0}, {"t1", 10.0}, {"dt", 0.1}, {"saveat", 0.5}}}};

  REQUIRE_NOTHROW(from_json<Time>(j));
}

TEST_CASE("[time][positivity] from_json_time_error_message_names_saveat") {
  json j = {{"timestepping", {{"t0", 0.0}, {"t1", 10.0}, {"dt", 0.1}}}};

  try {
    (void)from_json<Time>(j);
    FAIL("Expected std::invalid_argument to be thrown");
  } catch (const std::invalid_argument &e) {
    std::string msg = e.what();
    REQUIRE(msg.find("saveat") != std::string::npos);
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

TEST_CASE("[time][positivity] "
          "from_json_time_with_flat_structure_accepts_non_positive_saveat") {
  json j = {{"t0", 0.0}, {"t1", 10.0}, {"dt", 0.1}, {"saveat", -0.5}};

  const Time time = from_json<Time>(j);
  REQUIRE(time.get_saveat() == -0.5);
}

TEST_CASE("[time][positivity] "
          "from_json_time_with_flat_structure_rejects_t1_less_than_t0") {
  json j = {{"t0", 8.0}, {"t1", 2.0}, {"dt", 0.1}, {"saveat", 1.0}};

  REQUIRE_THROWS_AS(from_json<Time>(j), std::invalid_argument);
}

TEST_CASE("[time][positivity] from_json_time_rejects_nan_dt") {
  json j = {
      {"timestepping", {{"t0", 0.0}, {"t1", 10.0}, {"dt", 0.1}, {"saveat", 1.0}}}};

  // Set dt to NaN
  std::vector<uint8_t> nan_bytes = {0x00, 0x00, 0x00, 0x00,
                                    0x00, 0x00, 0xf8, 0x7f}; // double NaN
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
  std::vector<uint8_t> nan_bytes = {0x00, 0x00, 0x00, 0x00,
                                    0x00, 0x00, 0xf8, 0x7f}; // double NaN
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
