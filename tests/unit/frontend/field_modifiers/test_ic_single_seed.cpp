// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json_field_modifiers.hpp>
#include <openpfc/frontend/ui/lattice_json.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/simulation/initial_conditions/single_seed.hpp>
#include <openpfc/kernel/simulation/lattice.hpp>

using namespace pfc;
using Catch::Approx;

TEST_CASE("SingleSeed - Parameter Access", "[ic_single_seed]") {
  SingleSeed seed;

  SECTION("Set and get amplitude") {
    seed.set_amplitude(0.3);
    REQUIRE(seed.get_amplitude() == Approx(0.3));
  }

  SECTION("Set and get density") {
    seed.set_density(0.7);
    REQUIRE(seed.get_density() == Approx(0.7));
  }
}

TEST_CASE("SingleSeed - Field Application", "[ic_single_seed]") {
  auto domain = pfc::domain::create(pfc::GridSize({16, 16, 16}),
                                    pfc::PhysicalOrigin({-128.0, -128.0, -128.0}),
                                    pfc::GridSpacing({16.0, 16.0, 16.0}));
  auto box = pfc::domain::index_box(domain);
  std::vector<double> psi(static_cast<size_t>(box.count()), 0.0);

  SingleSeed seed;
  seed.set_amplitude(0.2);
  seed.set_density(0.5);

  SECTION("Apply to field") {
    seed.apply(psi, domain, box);

    bool has_nonzero = false;
    for (const auto &value : psi) {
      if (value != 0.0) {
        has_nonzero = true;
        break;
      }
    }
    REQUIRE(has_nonzero);
  }

  SECTION("Field values inside seed") {
    seed.apply(psi, domain, box);

    double max_expected = seed.get_density() + 12.0 * seed.get_amplitude();
    double min_expected = seed.get_density() - 12.0 * seed.get_amplitude();

    bool values_in_range = true;
    for (const auto &value : psi) {
      if (value != 0.0) {
        values_in_range &= value >= min_expected;
        values_in_range &= value <= max_expected;
      }
    }
    REQUIRE(values_in_range);
  }
}

TEST_CASE("SingleSeed - Apply on named density field", "[ic_single_seed]") {
  auto domain = pfc::domain::create(pfc::GridSize({8, 8, 8}),
                                    pfc::PhysicalOrigin({-64.0, -64.0, -64.0}),
                                    pfc::GridSpacing({16.0, 16.0, 16.0}));
  auto box = pfc::domain::index_box(domain);
  std::vector<double> psi(static_cast<size_t>(box.count()), 0.0);

  SingleSeed seed;
  seed.set_field_name("density");
  seed.set_amplitude(0.1);
  seed.set_density(0.6);

  REQUIRE_NOTHROW(seed.apply(psi, domain, box));
  REQUIRE(psi.size() == static_cast<size_t>(box.count()));
}

TEST_CASE("SingleSeed - Field Name Assignment", "[ic_single_seed]") {
  SingleSeed seed;
  seed.set_field_name("custom_field");
  REQUIRE(seed.get_field_name() == "custom_field");
}

TEST_CASE("SingleSeed - lattice selects the wave set", "[ic_single_seed][lattice]") {
  // Even grid with a node at the origin: 16 cells of spacing 1 from -8.
  auto domain = pfc::domain::create(pfc::GridSize({16, 16, 16}),
                                    pfc::PhysicalOrigin({-8.0, -8.0, -8.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto box = pfc::domain::index_box(domain);
  const std::size_t origin_index = 8 + 16 * (8 + 16 * 8);
  std::vector<double> bcc(static_cast<size_t>(box.count()), 0.0);
  std::vector<double> fcc(static_cast<size_t>(box.count()), 0.0);

  SingleSeed seed;
  seed.set_amplitude(0.2);
  seed.set_density(-0.4);
  REQUIRE(seed.get_lattice() == Lattice::bcc);
  seed.apply(bcc, domain, box);
  seed.set_lattice(Lattice::fcc);
  REQUIRE(seed.get_lattice() == Lattice::fcc);
  seed.apply(fcc, domain, box);

  // Six BCC waves versus four FCC waves at the seed centre.
  REQUIRE(bcc[origin_index] == Approx(-0.4 + 12.0 * 0.2));
  REQUIRE(fcc[origin_index] == Approx(-0.4 + 8.0 * 0.2));
  double diff = 0.0;
  for (std::size_t i = 0; i < bcc.size(); ++i) diff += std::abs(bcc[i] - fcc[i]);
  REQUIRE(diff > 1.0);
}

TEST_CASE("SingleSeed - JSON lattice key is optional and fails closed",
          "[ic_single_seed][lattice]") {
  SingleSeed seed;
  pfc::ui::from_json(
      nlohmann::json{{"type", "single_seed"}, {"amp_eq", 0.2}, {"rho_seed", -0.4}},
      seed);
  REQUIRE(seed.get_lattice() == Lattice::bcc);
  pfc::ui::from_json(nlohmann::json{{"type", "single_seed"},
                                    {"amp_eq", 0.2},
                                    {"rho_seed", -0.4},
                                    {"lattice", "fcc"}},
                     seed);
  REQUIRE(seed.get_lattice() == Lattice::fcc);
  REQUIRE_THROWS_AS(pfc::ui::from_json(nlohmann::json{{"type", "single_seed"},
                                                      {"amp_eq", 0.2},
                                                      {"rho_seed", -0.4},
                                                      {"lattice", "hcp"}},
                                       seed),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::ui::from_json(nlohmann::json{{"type", "single_seed"},
                                                      {"amp_eq", 0.2},
                                                      {"rho_seed", -0.4},
                                                      {"lattice", 1}},
                                       seed),
                    std::invalid_argument);
}

TEST_CASE("Model lattice is handed to seeds that leave the key out",
          "[ic_single_seed][lattice]") {
  nlohmann::json settings = {
      {"initial_conditions",
       nlohmann::json::array(
           {{{"type", "constant"}, {"n0", -0.4}},
            {{"type", "single_seed"}, {"amp_eq", 0.2}, {"rho_seed", -0.4}},
            {{"type", "seed_grid"}, {"lattice", "fcc"}},
            {{"type", "random_seeds"}, {"lattice", "bcc"}}})}};

  SECTION("missing key inherits, matching key stays, constant untouched") {
    nlohmann::json s = settings;
    s["initial_conditions"].erase(3);
    pfc::ui::apply_model_lattice_to_initial_conditions(s, Lattice::fcc);
    REQUIRE_FALSE(s["initial_conditions"][0].contains("lattice"));
    REQUIRE(s["initial_conditions"][1]["lattice"] == "fcc");
    REQUIRE(s["initial_conditions"][2]["lattice"] == "fcc");
  }

  SECTION("a seed naming another lattice than the model is rejected") {
    nlohmann::json s = settings;
    REQUIRE_THROWS_AS(
        pfc::ui::apply_model_lattice_to_initial_conditions(s, Lattice::fcc),
        std::invalid_argument);
  }

  SECTION("no initial conditions is not an error") {
    nlohmann::json s = nlohmann::json::object();
    REQUIRE_NOTHROW(
        pfc::ui::apply_model_lattice_to_initial_conditions(s, Lattice::bcc));
  }
}
