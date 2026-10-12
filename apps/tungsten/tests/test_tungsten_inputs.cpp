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
#include <openpfc/frontend/ui/from_json_world_time.hpp>
#include <openpfc/frontend/ui/settings_loader.hpp>
#include <openpfc/kernel/data/domain.hpp>
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

/// Every shipped input in any format: inputs_json/, inputs_toml/ and the
/// stand-alone CUDA example next to them.
std::vector<std::filesystem::path> all_shipped_inputs() {
  std::vector<std::filesystem::path> out = shipped_inputs();
  const std::filesystem::path toml_dir{TUNGSTEN_INPUTS_TOML_DIR};
  for (const auto &entry : std::filesystem::directory_iterator(toml_dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".toml") {
      out.push_back(entry.path());
    }
  }
  out.emplace_back(TUNGSTEN_CUDA_EXAMPLE_JSON);
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

// A boundary condition placed outside the box changes nothing and still costs
// a full pass over the field every step. tungsten_single_seed once declared a
// fixed BC whose band lay beyond its center-origin domain. The input's box is
// kept and its grid coarsened, so the 1024^3 inputs stay cheap.
TEST_CASE("Every shipped tungsten boundary condition touches the box",
          "[tungsten][inputs]") {
  tungsten::register_catalog();

  for (const auto &path : all_shipped_inputs()) {
    INFO("input file: " << path.filename().string());
    const json cfg = pfc::ui::load_settings_file(path);
    if (!cfg.contains("boundary_conditions")) continue;

    const pfc::Domain full = pfc::ui::from_json<pfc::Domain>(cfg);
    const pfc::Int3 n = pfc::domain::get_size(full);
    const pfc::Int3 coarse{std::min(n[0], 64), std::min(n[1], 8), std::min(n[2], 8)};
    const pfc::Domain domain = pfc::domain::from_bounds(
        coarse, pfc::domain::lower(full), pfc::domain::upper(full));
    const pfc::Box3i box = pfc::domain::index_box(domain);

    for (const json &entry : cfg["boundary_conditions"]) {
      const std::string type = entry["type"].get<std::string>();
      INFO("boundary condition: " << type);
      auto bc = pfc::ui::create_field_modifier(type, entry);
      constexpr double untouched = 12345.0;
      std::vector<double> field(pfc::domain::get_total_size(domain), untouched);
      bc->apply(pfc::field::FieldOutput<double>(field), domain, box, 0.0);
      const auto changed = std::count_if(field.begin(), field.end(),
                                         [](double v) { return v != untouched; });
      REQUIRE(changed > 0);
    }
  }
}
