// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

// The provenance table is the classification, not a material validation.
// Presets enter through load_settings_file and AluminumPhysics::from_json.

#include <aluminum/aluminum_session.hpp>
#include <aluminum/coefficient_provenance.hpp>
#include <aluminum/fixed_bc.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/ui/settings_loader.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/simulation/initial_conditions/constant.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace {

const aluminum::DensityCoefficient &density(std::string_view key) {
  for (const aluminum::DensityCoefficient &rec : aluminum::kDensityCoefficients) {
    if (rec.key == key) return rec;
  }
  throw std::invalid_argument("aluminum coefficient: unknown density key");
}

void require_derived(const aluminum::AluminumParams &p) {
  for (const aluminum::DerivedCoefficient &rec : aluminum::kDerivedCoefficients) {
    REQUIRE_FALSE(rec.formula.empty());
    if (rec.key == "tau_const") {
      REQUIRE(p.tau_const == p.T_const / p.T0);
    } else if (rec.key == "q2_bar") {
      REQUIRE(p.q2_bar == p.q21_bar * p.tau_const + p.q20_bar);
    } else if (rec.key == "q2_bar_L") {
      REQUIRE(p.q2_bar_L == p.q2_bar);
    } else if (rec.key == "q3_bar") {
      REQUIRE(p.q3_bar == p.q31_bar * p.tau_const + p.q30_bar);
    } else if (rec.key == "q4_bar") {
      REQUIRE(p.q4_bar == p.q40_bar);
    } else if (rec.key == "m_xpos") {
      REQUIRE(p.m_xpos == p.x_initial);
    } else {
      FAIL("derived coefficient has no oracle: " << rec.key);
    }
  }
}

} // namespace

TEST_CASE("aluminum coefficient provenance matches the shipped presets",
          "[aluminum][provenance]") {
  for (const aluminum::ModelCoefficient &rec : aluminum::coefficient_provenance()) {
    REQUIRE_FALSE(rec.cites_external_source);
    REQUIRE_FALSE(rec.source.empty());
    REQUIRE_FALSE(rec.meaning.empty());
    REQUIRE(aluminum::coefficient_value(rec.key) == rec.value);
    if (rec.status == aluminum::CoefficientStatus::representative) {
      REQUIRE(rec.source == aluminum::kUncitedSource);
    }
  }
  for (const aluminum::DensityCoefficient &rec : aluminum::kDensityCoefficients) {
    REQUIRE_FALSE(rec.cites_external_source);
    REQUIRE(rec.source == aluminum::kUncitedSource);
  }

  const aluminum::ReducedUnitConvention units = aluminum::reduced_unit_convention();
  REQUIRE_FALSE(units.length_to_si);
  REQUIRE_FALSE(units.temperature_to_si);
  REQUIRE_FALSE(units.time_to_si);
  REQUIRE_FALSE(units.amplitude_to_si);
  REQUIRE_FALSE(units.claims_material_validation);
  REQUIRE(std::string(units.length).find("No conversion to metres") !=
          std::string::npos);
  REQUIRE(std::string(units.temperature).find("No conversion to kelvin") !=
          std::string::npos);
  REQUIRE(std::string(units.time).find("No conversion to seconds") !=
          std::string::npos);
  REQUIRE(std::string(units.amplitude).find("No conversion to a number density") !=
          std::string::npos);

  aluminum::AluminumSchemaValues defaults;
  nlohmann::json from_table = nlohmann::json::object();
  for (const aluminum::ModelCoefficient &rec : aluminum::coefficient_provenance()) {
    REQUIRE(aluminum::schema_field(defaults, rec.key) == rec.value);
    from_table[std::string(rec.key)] = rec.value;
  }
  const auto schema = aluminum::make_aluminum_schema();
  // Prevent the provenance table silently falling behind the actual schema.
  // Each schema.docs_table() row describes one registered model parameter.
  const std::string docs = schema.docs_table();
  std::istringstream documented(docs);
  std::string row;
  int registered = 0;
  while (std::getline(documented, row)) {
    if (row.starts_with("| ") && !row.starts_with("| Name |")) ++registered;
  }
  REQUIRE(registered == static_cast<int>(aluminum::coefficient_provenance().size()));
  for (const aluminum::ModelCoefficient &rec : aluminum::coefficient_provenance()) {
    REQUIRE(docs.find("| " + std::string(rec.key) + " |") != std::string::npos);
  }
  const aluminum::AluminumSchemaValues parsed_table = schema.parse(from_table);
  for (const aluminum::ModelCoefficient &rec : aluminum::coefficient_provenance()) {
    REQUIRE(aluminum::schema_field(parsed_table, rec.key) == rec.value);
  }

  const auto readme = std::filesystem::path{ALUMINUM_APP_DIR} / "README.md";
  std::ifstream readme_in(readme);
  REQUIRE(readme_in.good());
  const std::string readme_text((std::istreambuf_iterator<char>(readme_in)),
                                std::istreambuf_iterator<char>());
  REQUIRE(readme_text.find("coefficient_provenance()") != std::string::npos);
  REQUIRE(readme_text.find("reduced_unit_convention()") != std::string::npos);
  REQUIRE(readme_text.find("not a material validation") != std::string::npos);

  aluminum::register_catalog();
  const auto domain = pfc::domain::create({8, 8, 8});
  const auto box = pfc::Box3i::from_bounds({0, 0, 0}, {7, 7, 7});
  const std::filesystem::path app{ALUMINUM_APP_DIR};
  // Discover every shipped JSON case; a new maintained case must not evade
  // provenance validation by being absent from a hardcoded four-file list.
  std::vector<std::filesystem::path> presets;
  presets.push_back(app / "aluminumNew.json");
  presets.push_back(app / "aluminumNew.toml");
  for (const auto &item : std::filesystem::directory_iterator(app / "inputs_json")) {
    if (item.is_regular_file() && item.path().extension() == ".json")
      presets.push_back(item.path());
  }
  std::sort(presets.begin(), presets.end());
  REQUIRE(presets.size() >= 4);

  int seeded = 0;
  int fixed_bands = 0;
  int files_with_unread = 0;
  for (const std::filesystem::path &path : presets) {
    INFO(path.filename().string());
    const nlohmann::json settings = pfc::ui::load_settings_file(path);
    const nlohmann::json &params = settings.at("model").at("params");
    const auto phys = aluminum::AluminumPhysics<>::from_json(params, domain, box);
    require_derived(phys.params);
    REQUIRE(std::isfinite(phys.correlation_kernel(-1.0)));
    REQUIRE(phys.correlation_kernel(-1.0) > 0.0);
    REQUIRE(std::isfinite(phys.linear_symbol(-1.0)));
    const double deviation = phys.temperature_variation(phys.params.x_initial, 0.0);
    REQUIRE(std::isfinite(deviation));
    REQUIRE(phys.params.T_const + deviation >= phys.params.T_min);
    REQUIRE(phys.params.T_const + deviation <= phys.params.T_max);
    // The periodic unwrapping seam can shift a nonisothermal profile even
    // at x_initial. Only zero-gradient cases must return exactly zero.
    if (phys.params.G_grid == 0.0) REQUIRE(deviation == 0.0);

    for (const aluminum::ModelCoefficient &rec :
         aluminum::coefficient_provenance()) {
      if (!rec.locks_every_preset) continue;
      REQUIRE(aluminum::schema_field(phys.params, rec.key) == rec.value);
    }

    const nlohmann::json &dom = settings.at("domain");
    for (int axis = 0; axis < 3; ++axis) {
      const double spacing = (dom.at("upper").at(axis).get<double>() -
                              dom.at("lower").at(axis).get<double>()) /
                             dom.at("size").at(axis).get<double>();
      REQUIRE(spacing ==
              Catch::Approx(aluminum::reduced_fcc_cell_spacing()).margin(1e-9));
    }

    bool saw_seed = false;
    double seed_x0 = 0.0;
    for (const nlohmann::json &entry : settings.at("initial_conditions")) {
      const auto modifier =
          pfc::ui::create_field_modifier(entry.at("type").get<std::string>(), entry);
      if (const auto *constant =
              dynamic_cast<const pfc::Constant *>(modifier.get())) {
        REQUIRE(constant->get_density() == density("n0").value);
      }
      if (const auto *seed =
              dynamic_cast<const aluminum::SeedGridFCC *>(modifier.get())) {
        saw_seed = true;
        seed_x0 = seed->x0();
        REQUIRE(seed->density() == density("rho").value);
        // These seed-grid parameters are case-specific, not material constants.
        REQUIRE(entry.at("radius").get<double>() > 0.0);
        REQUIRE(std::isfinite(entry.at("amplitude").get<double>()));
        REQUIRE(entry.at("Ny").get<int>() >= 1);
        REQUIRE(entry.at("Nz").get<int>() >= 1);
        REQUIRE(std::isfinite(seed_x0));
      }
    }
    if (saw_seed) {
      ++seeded;
      // Only the isothermal demo ties its temperature station to the seed.
      // A pulled directional case is allowed to place it ahead of the grain.
      if (phys.params.G_grid == 0.0 && phys.params.V_grid == 0.0)
        REQUIRE(aluminum::schema_field(phys.params, "x_initial") == seed_x0);
      else
        REQUIRE(std::isfinite(aluminum::schema_field(phys.params, "x_initial")));
    } else {
      REQUIRE(aluminum::schema_field(phys.params, "x_initial") ==
              aluminum::coefficient_value("x_initial"));
    }

    if (settings.contains("boundary_conditions")) {
      for (const nlohmann::json &entry : settings["boundary_conditions"]) {
        const auto modifier = pfc::ui::create_field_modifier(
            entry.at("type").get<std::string>(), entry);
        if (const auto *fixed =
                dynamic_cast<const aluminum::FixedBC *>(modifier.get())) {
          ++fixed_bands;
          REQUIRE(fixed->rho_low() == density("rho_low").value);
          REQUIRE(fixed->rho_high() == density("rho_high").value);
        }
      }
    }

    const bool clean_preset = path.filename() == "aluminumNew.json" ||
                              path.filename() == "aluminumNew.toml";
    bool unread = false;
    for (const std::string_view key : aluminum::kUnreadModelParamKeys) {
      const bool present = params.contains(std::string(key));
      if (clean_preset) REQUIRE_FALSE(present);
      unread = unread || present;
    }
    if (unread) {
      ++files_with_unread;
      nlohmann::json stripped = params;
      for (const std::string_view key : aluminum::kUnreadModelParamKeys) {
        stripped.erase(std::string(key));
      }
      const auto bare =
          aluminum::AluminumPhysics<>::from_json(stripped, domain, box);
      for (const aluminum::ModelCoefficient &rec :
           aluminum::coefficient_provenance()) {
        REQUIRE(aluminum::schema_field(phys.params, rec.key) ==
                aluminum::schema_field(bare.params, rec.key));
      }
      if (params.contains("n0"))
        REQUIRE(params.at("n0").get<double>() == density("n0").value);
      if (params.contains("n_sol")) {
        REQUIRE(params.at("n_sol").get<double>() == density("rho").value);
      }
      if (params.contains("n_vap")) {
        REQUIRE(params.at("n_vap").get<double>() == density("rho_low").value);
      }
    }
  }
  REQUIRE(seeded >= 1);
  REQUIRE(fixed_bands >= 1);
  REQUIRE(files_with_unread >= 1);
}

// Some shipped modifiers call MPI_Comm_rank during construction. Use the
// same explicit MPI lifecycle as aluminumTest and the crystal-observable test.
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int status = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return status;
}
