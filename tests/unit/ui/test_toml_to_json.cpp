// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_toml_to_json.cpp
 * @brief Unit tests for TOML to JSON conversion utility
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <openpfc/utils/toml_to_json.hpp>
#include <sstream>

using namespace pfc::utils;
using Catch::Matchers::WithinRel;

TEST_CASE("toml_to_json converts basic types", "[toml][json]") {

  SECTION("string values") {
    auto toml_data = toml::parse(R"(
      name = "test"
      path = "/some/path"
    )");

    auto json = toml_to_json(toml_data);

    REQUIRE(json["name"] == "test");
    REQUIRE(json["path"] == "/some/path");
  }

  SECTION("integer values") {
    auto toml_data = toml::parse(R"(
      count = 42
      negative = -10
      zero = 0
    )");

    auto json = toml_to_json(toml_data);

    REQUIRE(json["count"] == 42);
    REQUIRE(json["negative"] == -10);
    REQUIRE(json["zero"] == 0);
  }

  SECTION("floating point values") {
    auto toml_data = toml::parse(R"(
      pi = 3.14159
      negative = -2.5
      scientific = 1.23e-4
    )");

    auto json = toml_to_json(toml_data);

    REQUIRE_THAT(json["pi"].get<double>(), WithinRel(3.14159, 1e-10));
    REQUIRE_THAT(json["negative"].get<double>(), WithinRel(-2.5, 1e-10));
    REQUIRE_THAT(json["scientific"].get<double>(), WithinRel(1.23e-4, 1e-10));
  }

  SECTION("boolean values") {
    auto toml_data = toml::parse(R"(
      enabled = true
      disabled = false
    )");

    auto json = toml_to_json(toml_data);

    REQUIRE(json["enabled"] == true);
    REQUIRE(json["disabled"] == false);
  }
}

TEST_CASE("toml_to_json converts arrays", "[toml][json]") {

  SECTION("array of integers") {
    auto toml_data = toml::parse(R"(
      numbers = [1, 2, 3, 4, 5]
    )");

    auto json = toml_to_json(toml_data);

    REQUIRE(json["numbers"].is_array());
    REQUIRE(json["numbers"].size() == 5);
    REQUIRE(json["numbers"][0] == 1);
    REQUIRE(json["numbers"][4] == 5);
  }

  SECTION("array of strings") {
    auto toml_data = toml::parse(R"(
      names = ["alice", "bob", "charlie"]
    )");

    auto json = toml_to_json(toml_data);

    REQUIRE(json["names"].is_array());
    REQUIRE(json["names"].size() == 3);
    REQUIRE(json["names"][0] == "alice");
    REQUIRE(json["names"][2] == "charlie");
  }

  SECTION("array of floats") {
    auto toml_data = toml::parse(R"(
      values = [1.1, 2.2, 3.3]
    )");

    auto json = toml_to_json(toml_data);

    REQUIRE(json["values"].is_array());
    REQUIRE_THAT(json["values"][0].get<double>(), WithinRel(1.1, 1e-10));
    REQUIRE_THAT(json["values"][2].get<double>(), WithinRel(3.3, 1e-10));
  }

  SECTION("empty array") {
    auto toml_data = toml::parse(R"(
      empty = []
    )");

    auto json = toml_to_json(toml_data);

    REQUIRE(json["empty"].is_array());
    REQUIRE(json["empty"].empty());
  }
}

TEST_CASE("toml_to_json converts nested tables", "[toml][json]") {

  SECTION("simple nested table") {
    auto toml_data = toml::parse(R"(
      [model]
      name = "tungsten"

      [model.params]
      n0 = -0.10
      T = 3300.0
    )");

    auto json = toml_to_json(toml_data);

    REQUIRE(json["model"]["name"] == "tungsten");
    REQUIRE_THAT(json["model"]["params"]["n0"].get<double>(),
                 WithinRel(-0.10, 1e-10));
    REQUIRE_THAT(json["model"]["params"]["T"].get<double>(),
                 WithinRel(3300.0, 1e-10));
  }

  SECTION("deeply nested tables") {
    auto toml_data = toml::parse(R"(
      [a]
      [a.b]
      [a.b.c]
      value = 123
    )");

    auto json = toml_to_json(toml_data);

    REQUIRE(json["a"]["b"]["c"]["value"] == 123);
  }
}

TEST_CASE("toml_to_json converts array of tables", "[toml][json]") {

  SECTION("simple array of tables") {
    auto toml_data = toml::parse(R"(
      [[fields]]
      name = "psi"
      data = "/path/to/psi.bin"

      [[fields]]
      name = "psiMF"
      data = "/path/to/psimf.bin"
    )");

    auto json = toml_to_json(toml_data);

    REQUIRE(json["fields"].is_array());
    REQUIRE(json["fields"].size() == 2);
    REQUIRE(json["fields"][0]["name"] == "psi");
    REQUIRE(json["fields"][0]["data"] == "/path/to/psi.bin");
    REQUIRE(json["fields"][1]["name"] == "psiMF");
  }

  SECTION("array of tables with different keys") {
    auto toml_data = toml::parse(R"(
      [[initial_conditions]]
      target = "psi"
      type = "constant"
      n0 = -0.4

      [[initial_conditions]]
      target = "psi"
      type = "single_seed"
      amp_eq = 0.215936
      rho_seed = -0.047
    )");

    auto json = toml_to_json(toml_data);

    REQUIRE(json["initial_conditions"].is_array());
    REQUIRE(json["initial_conditions"].size() == 2);
    REQUIRE(json["initial_conditions"][0]["type"] == "constant");
    REQUIRE_THAT(json["initial_conditions"][0]["n0"].get<double>(),
                 WithinRel(-0.4, 1e-10));
    REQUIRE(json["initial_conditions"][1]["type"] == "single_seed");
    REQUIRE_THAT(json["initial_conditions"][1]["amp_eq"].get<double>(),
                 WithinRel(0.215936, 1e-10));
  }
}

TEST_CASE("toml_to_json handles TOML-specific date/time types", "[toml][json]") {

  SECTION("date value") {
    auto toml_data = toml::parse(R"(
      start_date = 2025-01-15
    )");

    auto json = toml_to_json(toml_data);

    // Dates are converted to ISO 8601 string format
    REQUIRE(json["start_date"].is_string());
    REQUIRE(json["start_date"] == "2025-01-15");
  }

  SECTION("time value") {
    auto toml_data = toml::parse(R"(
      start_time = 14:30:00
    )");

    auto json = toml_to_json(toml_data);

    // Times are converted to ISO 8601 string format
    REQUIRE(json["start_time"].is_string());
    REQUIRE(json["start_time"] == "14:30:00");
  }

  SECTION("datetime value") {
    auto toml_data = toml::parse(R"(
      timestamp = 2025-01-15T14:30:00
    )");

    auto json = toml_to_json(toml_data);

    // Datetimes are converted to ISO 8601 string format
    REQUIRE(json["timestamp"].is_string());
    REQUIRE(json["timestamp"] == "2025-01-15T14:30:00");
  }
}

TEST_CASE("toml_to_json handles mixed arrays", "[toml][json]") {
  // Note: TOML 1.0 allows mixed-type arrays
  SECTION("array with nested arrays") {
    auto toml_data = toml::parse(R"(
      matrix = [[1, 2], [3, 4]]
    )");

    auto json = toml_to_json(toml_data);

    REQUIRE(json["matrix"].is_array());
    REQUIRE(json["matrix"].size() == 2);
    REQUIRE(json["matrix"][0].is_array());
    REQUIRE(json["matrix"][0][0] == 1);
    REQUIRE(json["matrix"][1][1] == 4);
  }
}

TEST_CASE("toml_to_json handles OpenPFC configuration patterns", "[toml][json]") {
  // Test a realistic OpenPFC configuration snippet
  auto toml_data = toml::parse(R"(
    [model]
    name = "tungsten"

    [model.params]
    n0 = -0.10
    n_sol = -0.047
    T = 3300.0
    alpha = 0.50

    [domain]
    Lx = 32
    Ly = 32
    Lz = 32
    dx = 1.1107207345395915
    dy = 1.1107207345395915
    dz = 1.1107207345395915
    origin = "center"

    [timestepping]
    t0 = 0.0
    t1 = 10.0
    dt = 1.0
    saveat = 1.0

    [[fields]]
    name = "psi"
    data = "/output/psi_%d.bin"

    [[initial_conditions]]
    target = "psi"
    type = "constant"
    n0 = -0.4

    [[boundary_conditions]]
    target = "psi"
    type = "fixed"
    rho_low = -0.464
    rho_high = -0.100
  )");

  auto json = toml_to_json(toml_data);

  // Verify model section
  REQUIRE(json["model"]["name"] == "tungsten");
  REQUIRE_THAT(json["model"]["params"]["n0"].get<double>(), WithinRel(-0.10, 1e-10));
  REQUIRE_THAT(json["model"]["params"]["T"].get<double>(), WithinRel(3300.0, 1e-10));

  // Verify domain section
  REQUIRE(json["domain"]["Lx"] == 32);
  REQUIRE(json["domain"]["origin"] == "center");

  // Verify timestepping section
  REQUIRE_THAT(json["timestepping"]["t0"].get<double>(), WithinRel(0.0, 1e-10));
  REQUIRE_THAT(json["timestepping"]["dt"].get<double>(), WithinRel(1.0, 1e-10));

  // Verify array of tables
  REQUIRE(json["fields"].is_array());
  REQUIRE(json["fields"][0]["name"] == "psi");

  REQUIRE(json["initial_conditions"].is_array());
  REQUIRE(json["initial_conditions"][0]["type"] == "constant");

  REQUIRE(json["boundary_conditions"].is_array());
  REQUIRE(json["boundary_conditions"][0]["type"] == "fixed");
}
