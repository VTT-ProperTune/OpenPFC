// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <openpfc/ui.hpp>

using json = nlohmann::json;
using namespace pfc;

TEST_CASE("World validation with wrong type for Lx",
          "[ui][integration][validation]") {
  json j = {{"Lx", 256.5}, // Float instead of int!
            {"Ly", 256},   {"Lz", 256}, {"dx", 1.0},
            {"dy", 1.0},   {"dz", 1.0}, {"origo", "center"}};

  try {
    [[maybe_unused]] auto world = ui::from_json<World>(j);
    FAIL("Should have thrown exception");
  } catch (const std::invalid_argument &e) {
    std::string msg(e.what());
    REQUIRE(msg.find("Lx") != std::string::npos);
    REQUIRE(msg.find("integer") != std::string::npos);
    REQUIRE(msg.find("256.5") != std::string::npos);
    REQUIRE(msg.find("Description") != std::string::npos);
    REQUIRE(msg.find("Expected") != std::string::npos);
    REQUIRE(msg.find("Got") != std::string::npos);
    REQUIRE(msg.find("Example") != std::string::npos);
  }
}

TEST_CASE("World validation with invalid origo value",
          "[ui][integration][validation]") {
  json j = {
      {"Lx", 256}, {"Ly", 256}, {"Lz", 256},        {"dx", 1.0},
      {"dy", 1.0}, {"dz", 1.0}, {"origo", "centre"} // British spelling!
  };

  try {
    [[maybe_unused]] auto world = ui::from_json<World>(j);
    FAIL("Should have thrown exception");
  } catch (const std::invalid_argument &e) {
    std::string msg(e.what());
    REQUIRE(msg.find("origo") != std::string::npos);
    REQUIRE(msg.find("Valid options") != std::string::npos);
    REQUIRE(msg.find("'center'") != std::string::npos);
    REQUIRE(msg.find("'corner'") != std::string::npos);
    REQUIRE(msg.find("centre") != std::string::npos);
  }
}

TEST_CASE("World validation with missing Ly field",
          "[ui][integration][validation]") {
  json j = {{"Lx", 256},
            // Ly is missing!
            {"Lz", 256},
            {"dx", 1.0},
            {"dy", 1.0},
            {"dz", 1.0},
            {"origo", "center"}};

  try {
    [[maybe_unused]] auto world = ui::from_json<World>(j);
    FAIL("Should have thrown exception");
  } catch (const std::invalid_argument &e) {
    std::string msg(e.what());
    REQUIRE(msg.find("Ly") != std::string::npos);
    REQUIRE(msg.find("missing") != std::string::npos);
    REQUIRE(msg.find("grid points in Y direction") != std::string::npos);
  }
}

TEST_CASE("World validation with wrong type for dx",
          "[ui][integration][validation]") {
  json j = {{"Lx", 256},   {"Ly", 256}, {"Lz", 256},
            {"dx", "1.0"}, // String instead of float!
            {"dy", 1.0},   {"dz", 1.0}, {"origo", "center"}};

  try {
    [[maybe_unused]] auto world = ui::from_json<World>(j);
    FAIL("Should have thrown exception");
  } catch (const std::invalid_argument &e) {
    std::string msg(e.what());
    REQUIRE(msg.find("dx") != std::string::npos);
    REQUIRE(msg.find("grid spacing") != std::string::npos);
    REQUIRE(msg.find("float") != std::string::npos);
  }
}

TEST_CASE("Unknown field modifier type error", "[ui][integration][validation]") {
  json params = {
      {"type", "random_seed"} // Typo: should be "random_seeds"
  };

  try {
    auto modifier = ui::create_field_modifier("random_seed", params);
    FAIL("Should have thrown exception");
  } catch (const std::invalid_argument &e) {
    std::string msg(e.what());
    REQUIRE(msg.find("Unknown") != std::string::npos);
    REQUIRE(msg.find("random_seed") != std::string::npos);
    REQUIRE(msg.find("Valid types") != std::string::npos);
    REQUIRE(msg.find("constant") != std::string::npos);
    REQUIRE(msg.find("single_seed") != std::string::npos);
    REQUIRE(msg.find("random_seeds") != std::string::npos);
    REQUIRE(msg.find("fixed") != std::string::npos);
    REQUIRE(msg.find("moving") != std::string::npos);
  }
}

TEST_CASE("Error messages are concise", "[ui][integration][validation]") {
  json j = {{"Lx", 256.5}, {"Ly", 256}, {"Lz", 256},        {"dx", 1.0},
            {"dy", 1.0},   {"dz", 1.0}, {"origo", "center"}};

  try {
    [[maybe_unused]] auto world = ui::from_json<World>(j);
    FAIL("Should have thrown exception");
  } catch (const std::invalid_argument &e) {
    std::string msg(e.what());
    // Count newlines - should be < 10 lines
    int lines = 1 + std::count(msg.begin(), msg.end(), '\n');
    REQUIRE(lines <= 10);
  }
}
