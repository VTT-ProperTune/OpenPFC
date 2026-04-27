// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/ui/json_helpers.hpp>

using json = nlohmann::json;
using pfc::ui::get_json_value;

TEST_CASE("get_json_value prefers top-level keys", "[ui][json]") {
  const json config = {
      {"Lx", 64},
      {"domain", {{"Lx", 128}}},
  };

  REQUIRE(get_json_value(config, "Lx") == 64);
  REQUIRE(get_json_value(config, "Lx", "domain") == 64);
}

TEST_CASE("get_json_value reads keys from explicit sections", "[ui][json]") {
  const json config = {
      {"domain", {{"Lx", 128}, {"dx", 0.5}}},
      {"timestepping", {{"dt", 0.01}}},
  };

  REQUIRE(get_json_value(config, "Lx", "domain") == 128);
  REQUIRE(get_json_value(config, "dx", "domain") == 0.5);
  REQUIRE(get_json_value(config, "dt", "timestepping") == 0.01);
}

TEST_CASE("get_json_value searches common sections without a section",
          "[ui][json]") {
  const json config = {
      {"domain", {{"Lz", 32}}},
      {"timestepping", {{"saveat", 10}}},
  };

  REQUIRE(get_json_value(config, "Lz") == 32);
  REQUIRE(get_json_value(config, "saveat") == 10);
}

TEST_CASE("get_json_value returns null for missing keys", "[ui][json]") {
  const json config = {
      {"domain", {{"Lx", 128}}},
      {"timestepping", {{"dt", 0.01}}},
  };

  const auto missing = get_json_value(config, "Ly");
  REQUIRE(missing.is_null());
  REQUIRE(get_json_value(config, "dt", "domain").is_null());
}
