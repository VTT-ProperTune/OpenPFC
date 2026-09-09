// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_tungsten_inputs.cpp
 * @brief Every shipped JSON input must actually load.
 *
 * @details
 * `tungsten_moving_bc_options.json` sat in the tree with
 * `"initial_position": "end"` where `MovingBC` requires a numeric `xpos`.
 * Running it aborted on the first line of boundary-condition wiring:
 *
 *     Invalid JSON input: missing or invalid 'xpos' field.
 *
 * Nothing caught it, because nothing here ever loaded the shipped inputs --
 * the tests build their JSON inline. A file a reader is invited to run is part
 * of the product, so this walks every input in `inputs_json/` and constructs
 * what it declares: the model parameters through the tungsten schema, and each
 * initial and boundary condition through the same catalog the application
 * uses.
 *
 * It stops short of stepping a simulation. The point is to catch a key that
 * was renamed, dropped, or never read, which is cheap to check and is what
 * actually rots; a physics regression is a different test's job.
 */

#define CATCH_CONFIG_ENABLE_ALL_STRINGMAKERS
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <tungsten/tungsten_physics.hpp>
#include <tungsten/tungsten_session.hpp>

namespace {

using json = nlohmann::json;

/// Shipped inputs, sorted so a failure names the same file on every machine.
std::vector<std::filesystem::path> shipped_inputs() {
  std::vector<std::filesystem::path> out;
  const std::filesystem::path dir{TUNGSTEN_INPUTS_JSON_DIR};
  for (const auto &entry : std::filesystem::directory_iterator(dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".json") {
      out.push_back(entry.path());
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

} // namespace

TEST_CASE("Every shipped tungsten JSON input loads", "[tungsten][inputs]") {
  tungsten::register_catalog();

  const auto inputs = shipped_inputs();
  INFO("inputs directory: " << TUNGSTEN_INPUTS_JSON_DIR);
  REQUIRE_FALSE(inputs.empty());

  for (const auto &path : inputs) {
    const std::string name = path.filename().string();
    INFO("input file: " << name);

    std::ifstream in(path);
    REQUIRE(in.good());
    json cfg;
    REQUIRE_NOTHROW(in >> cfg);

    // Model parameters, through the schema the application parses them with.
    if (cfg.contains("model") && cfg["model"].contains("params")) {
      const json &params = cfg["model"]["params"];
      INFO("section: model.params");
      REQUIRE_NOTHROW(tungsten::make_tungsten_schema().parse(params));
    }

    // Initial and boundary conditions, through the same catalog the session
    // uses -- this is where a renamed or missing key shows up.
    for (const char *section : {"initial_conditions", "boundary_conditions"}) {
      if (!cfg.contains(section)) continue;
      for (const json &entry : cfg[section]) {
        REQUIRE(entry.contains("type"));
        const std::string type = entry["type"].get<std::string>();
        INFO("section: " << section << ", type: " << type);
        REQUIRE_NOTHROW(pfc::ui::create_field_modifier(type, entry));
      }
    }
  }
}
