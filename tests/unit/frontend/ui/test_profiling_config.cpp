// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/ui/app.hpp>
#include <sstream>

using json = nlohmann::json;
using pfc::ui::warn_unknown_profiling_keys;

TEST_CASE("warn_unknown_profiling_keys ignores known keys", "[ui][profiling]") {
  const json profiling = {
      {"enabled", true},       {"format", "json"},
      {"output", "profile"},   {"memory_samples", true},
      {"print_report", false}, {"regions", json::array({"gradient/custom"})},
      {"run_id", "run-1"},     {"export_metadata", {{"case", "unit"}}},
  };
  std::ostringstream warnings;

  warn_unknown_profiling_keys(profiling, warnings);

  REQUIRE(warnings.str().empty());
}

TEST_CASE("warn_unknown_profiling_keys reports unknown keys", "[ui][profiling]") {
  const json profiling = {
      {"enabled", true},
      {"enable", true},
      {"outptu", "profile"},
  };
  std::ostringstream warnings;

  warn_unknown_profiling_keys(profiling, warnings);

  const auto text = warnings.str();
  REQUIRE(text.find("enable") != std::string::npos);
  REQUIRE(text.find("outptu") != std::string::npos);
  REQUIRE(text.find("enabled") == std::string::npos);
}
