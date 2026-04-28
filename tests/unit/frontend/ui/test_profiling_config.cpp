// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/ui/app_profiling.hpp>
#include <sstream>
#include <string>

using json = nlohmann::json;
using pfc::ui::list_unknown_profiling_keys;

TEST_CASE("warn_unknown_profiling_keys ignores known keys", "[ui][profiling]") {
  const json profiling = {
      {"enabled", true},       {"format", "json"},
      {"output", "profile"},   {"memory_samples", true},
      {"print_report", false}, {"regions", json::array({"gradient/custom"})},
      {"run_id", "run-1"},     {"export_metadata", {{"case", "unit"}}},
  };

  REQUIRE(list_unknown_profiling_keys(profiling).empty());
}

TEST_CASE("warn_unknown_profiling_keys reports unknown keys", "[ui][profiling]") {
  const json profiling = {
      {"enabled", true},
      {"enable", true},
      {"outptu", "profile"},
  };

  const auto warnings = list_unknown_profiling_keys(profiling);
  REQUIRE(warnings.size() == 2);
  std::ostringstream joined;
  for (const auto &w : warnings) {
    joined << w;
  }
  const std::string text = joined.str();
  REQUIRE(text.find("key 'enable'") != std::string::npos);
  REQUIRE(text.find("outptu") != std::string::npos);
  REQUIRE(text.find("key 'enabled'") == std::string::npos);
}
