// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <openpfc/ui_errors.hpp>

using json = nlohmann::json;
using namespace pfc::ui;

TEST_CASE("format_config_error with missing field", "[ui][errors]") {
  auto msg = format_config_error("Lx", "number of grid points in X direction",
                                 "positive integer", "missing");

  REQUIRE(msg.find("Field 'Lx' is missing") != std::string::npos);
  REQUIRE(msg.find("positive integer") != std::string::npos);
  REQUIRE(msg.find("number of grid points") != std::string::npos);
}

TEST_CASE("format_config_error with invalid type", "[ui][errors]") {
  auto msg = format_config_error("Lx", "number of grid points", "integer",
                                 "256.5 (type: float)");

  REQUIRE(msg.find("has invalid value") != std::string::npos);
  REQUIRE(msg.find("256.5") != std::string::npos);
  REQUIRE(msg.find("integer") != std::string::npos);
}

TEST_CASE("format_config_error with valid options", "[ui][errors]") {
  auto msg = format_config_error("origo", "coordinate origin", "string",
                                 "\"centre\"", {"center", "corner"});

  REQUIRE(msg.find("Valid options") != std::string::npos);
  REQUIRE(msg.find("'center'") != std::string::npos);
  REQUIRE(msg.find("'corner'") != std::string::npos);
}

TEST_CASE("format_config_error with example", "[ui][errors]") {
  auto msg = format_config_error("Lx", "grid points", "integer", "missing", {},
                                 "\"Lx\": 256");

  REQUIRE(msg.find("Example: \"Lx\": 256") != std::string::npos);
}

TEST_CASE("format_config_error is concise", "[ui][errors]") {
  // Test that error messages are not too long (< 10 lines typical)
  auto msg = format_config_error("Lx", "number of grid points", "positive integer",
                                 "256.5 (type: float)", {}, "\"Lx\": 256");

  // Count newlines
  int lines = 1 + std::count(msg.begin(), msg.end(), '\n');
  REQUIRE(lines <= 10);
}

TEST_CASE("get_json_value_string handles missing field", "[ui][errors]") {
  json j = {{"foo", 42}};
  auto result = get_json_value_string(j, "bar");
  REQUIRE(result == "missing");
}

TEST_CASE("get_json_value_string shows type for float", "[ui][errors]") {
  json j = {{"num", 42.5}};
  auto result = get_json_value_string(j, "num");
  REQUIRE(result.find("42.5") != std::string::npos);
  REQUIRE(result.find("float") != std::string::npos);
}

TEST_CASE("get_json_value_string shows type for integer", "[ui][errors]") {
  json j = {{"num", 42}};
  auto result = get_json_value_string(j, "num");
  REQUIRE(result.find("42") != std::string::npos);
  REQUIRE(result.find("integer") != std::string::npos);
}

TEST_CASE("get_json_value_string shows type for string", "[ui][errors]") {
  json j = {{"str", "hello"}};
  auto result = get_json_value_string(j, "str");
  REQUIRE(result.find("hello") != std::string::npos);
  REQUIRE(result.find("string") != std::string::npos);
}

TEST_CASE("get_json_value_string shows type for boolean", "[ui][errors]") {
  json j = {{"flag", true}};
  auto result = get_json_value_string(j, "flag");
  REQUIRE(result.find("true") != std::string::npos);
  REQUIRE(result.find("boolean") != std::string::npos);
}

TEST_CASE("get_json_value_string shows type for null", "[ui][errors]") {
  json j = {{"val", nullptr}};
  auto result = get_json_value_string(j, "val");
  REQUIRE(result.find("null") != std::string::npos);
}

TEST_CASE("get_json_value_string shows type for array", "[ui][errors]") {
  json j = {{"arr", {1, 2, 3}}};
  auto result = get_json_value_string(j, "arr");
  REQUIRE(result.find("array") != std::string::npos);
}

TEST_CASE("get_json_value_string shows type for object", "[ui][errors]") {
  json j = {{"obj", {{"key", "value"}}}};
  auto result = get_json_value_string(j, "obj");
  REQUIRE(result.find("object") != std::string::npos);
}

TEST_CASE("format_unknown_modifier_error lists valid types", "[ui][errors]") {
  auto msg = format_unknown_modifier_error("random_seed");

  REQUIRE(msg.find("Unknown") != std::string::npos);
  REQUIRE(msg.find("'random_seed'") != std::string::npos);
  REQUIRE(msg.find("Valid types") != std::string::npos);
  REQUIRE(msg.find("constant") != std::string::npos);
  REQUIRE(msg.find("single_seed") != std::string::npos);
}

TEST_CASE("format_unknown_modifier_error uses custom context", "[ui][errors]") {
  auto msg = format_unknown_modifier_error("invalid", "initial condition");

  REQUIRE(msg.find("initial condition") != std::string::npos);
  REQUIRE(msg.find("'invalid'") != std::string::npos);
}

TEST_CASE("list_valid_field_modifiers returns non-empty", "[ui][errors]") {
  auto types = list_valid_field_modifiers();
  REQUIRE(!types.empty());
  REQUIRE(types.size() >= 6); // At least constant, single_seed, random_seeds,
                              // seed_grid, from_file, fixed, moving
}

TEST_CASE("list_valid_field_modifiers includes initial conditions", "[ui][errors]") {
  auto types = list_valid_field_modifiers();
  REQUIRE(std::find(types.begin(), types.end(), "constant") != types.end());
  REQUIRE(std::find(types.begin(), types.end(), "single_seed") != types.end());
  REQUIRE(std::find(types.begin(), types.end(), "random_seeds") != types.end());
  REQUIRE(std::find(types.begin(), types.end(), "seed_grid") != types.end());
  REQUIRE(std::find(types.begin(), types.end(), "from_file") != types.end());
}

TEST_CASE("list_valid_field_modifiers includes boundary conditions",
          "[ui][errors]") {
  auto types = list_valid_field_modifiers();
  REQUIRE(std::find(types.begin(), types.end(), "fixed") != types.end());
  REQUIRE(std::find(types.begin(), types.end(), "moving") != types.end());
}
