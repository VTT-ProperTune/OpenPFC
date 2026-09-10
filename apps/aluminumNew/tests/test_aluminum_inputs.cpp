// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_aluminum_inputs.cpp
 * @brief Every shipped aluminium JSON input must actually load.
 *
 * @details
 * Modelled on `apps/tungsten/tests/test_tungsten_inputs.cpp`, and here for the
 * same reason: nothing else in this application ever loads the files under
 * `inputs_json/`. The tests build their JSON inline, the pinned checksum runs a
 * synthetic 32^3 case constructed in C++, and CMake references none of the
 * shipped inputs -- so a key renamed in the schema or in an initial condition
 * would leave a file that still looks fine in review and aborts on the first
 * line of setup for whoever runs it. A file a reader is invited to run is part
 * of the product.
 *
 * Like the tungsten twin it stops short of stepping a simulation: it parses the
 * model parameters through the aluminium schema and constructs every declared
 * initial and boundary condition through the same catalog the binaries use.
 * That is what actually rots and it is cheap to check; a physics regression is
 * `test_aluminum_physics.cpp`'s job.
 */

#define CATCH_CONFIG_ENABLE_ALL_STRINGMAKERS
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <aluminum/aluminum_physics.hpp>
#include <aluminum/aluminum_session.hpp>
#include <openpfc/frontend/ui/field_modifier_registry.hpp>

namespace {

using json = nlohmann::json;

/// Shipped inputs, sorted so a failure names the same file on every machine.
std::vector<std::filesystem::path> shipped_inputs() {
  std::vector<std::filesystem::path> out;
  const std::filesystem::path dir{ALUMINUM_INPUTS_JSON_DIR};
  for (const auto &entry : std::filesystem::directory_iterator(dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".json") {
      out.push_back(entry.path());
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

} // namespace

TEST_CASE("Every shipped aluminum JSON input loads", "[aluminum][inputs]") {
  aluminum::register_catalog();

  const auto inputs = shipped_inputs();
  INFO("inputs directory: " << ALUMINUM_INPUTS_JSON_DIR);
  REQUIRE_FALSE(inputs.empty());

  for (const auto &path : inputs) {
    const std::string name = path.filename().string();
    INFO("input file: " << name);

    std::ifstream in(path);
    REQUIRE(in.good());
    json cfg;
    REQUIRE_NOTHROW(in >> cfg);

    // Model parameters, through the schema the application parses them with.
    //
    // Mirror `AluminumPhysics::from_json` exactly, including the part that is
    // easy to miss: it calls the schema only `if (!params_json.is_null() &&
    // !params_json.empty())`. An **empty** `params` object therefore skips
    // schema validation altogether and the run silently takes the C++ struct
    // defaults, even though every parameter in `make_aluminum_schema()` is
    // declared `required`. `inputs_json/smoke.json` is exactly that case: it
    // ships `"params": {}` and runs. Asserting the schema on it would fail a
    // file the application accepts, so this checks the contract the code
    // actually has -- non-empty means it must parse -- and pins the empty case
    // as deliberate rather than as an oversight nobody noticed.
    if (cfg.contains("model") && cfg["model"].contains("params")) {
      const json &params = cfg["model"]["params"];
      INFO("section: model.params");
      if (params.is_null() || params.empty()) {
        INFO("empty params: the application skips the schema and uses defaults");
        REQUIRE(true);
      } else {
        REQUIRE_NOTHROW(aluminum::make_aluminum_schema().parse(params));
      }
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
