// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <stdexcept>
#include <type_traits>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <tungsten/fixed_bc.hpp>
#include <tungsten/moving_bc.hpp>
#include <tungsten/solidification_bc_json.hpp>

using namespace pfc;
using tungsten::FixedBC;
using tungsten::MovingBC;

static_assert(std::is_base_of_v<pfc::FieldModifier, tungsten::FixedBC>);
static_assert(std::is_base_of_v<pfc::FieldModifier, tungsten::MovingBC>);

TEST_CASE("FixedBC - Basic functionality", "[boundary_conditions][unit]") {
  SECTION("FixedBC can be constructed with default values") {
    FixedBC fixedBC;
    REQUIRE_NOTHROW(fixedBC);
  }

  SECTION("FixedBC can be constructed with parameters") {
    FixedBC fixedBC(-0.5, 0.5);
    REQUIRE_NOTHROW(fixedBC);
  }

  SECTION("FixedBC field name can be set and retrieved") {
    FixedBC fixedBC;
    fixedBC.set_field_name("psi");
    REQUIRE(fixedBC.get_field_name() == "psi");
  }

  SECTION("FixedBC rho values can be set") {
    FixedBC fixedBC;
    fixedBC.set_rho_low(-0.5);
    fixedBC.set_rho_high(0.5);
    REQUIRE_NOTHROW(fixedBC);
  }

  SECTION("FixedBC has correct modifier name") {
    FixedBC fixedBC;
    REQUIRE(fixedBC.get_modifier_name() == "FixedBC");
  }
}

TEST_CASE("FixedBC - apply method", "[boundary_conditions][unit]") {
  auto domain = pfc::domain::create(pfc::GridSize({100, 1, 1}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto box = pfc::domain::index_box(domain);
  std::vector<double> field_data(static_cast<size_t>(box.count()), 0.0);

  SECTION("FixedBC applies boundary smoothly") {
    FixedBC bc(-1.0, 1.0);
    bc.set_field_name("psi");
    REQUIRE_NOTHROW(bc.apply(field_data, domain, box));
    bool has_nonzero = false;
    for (const auto &val : field_data) {
      if (val != 0.0) {
        has_nonzero = true;
        break;
      }
    }
    REQUIRE(has_nonzero);
  }

  SECTION("FixedBC respects rho_low and rho_high parameters") {
    FixedBC bc;
    bc.set_rho_low(-2.5);
    bc.set_rho_high(3.5);
    bc.set_field_name("psi");
    REQUIRE_NOTHROW(bc.apply(field_data, domain, box));
    bool values_in_range = true;
    for (const auto &val : field_data) values_in_range &= val >= -2.6 && val <= 3.6;
    REQUIRE(values_in_range);
  }

  SECTION("FixedBC can be applied multiple times") {
    FixedBC bc(0.0, 1.0);
    bc.set_field_name("psi");
    REQUIRE_NOTHROW(bc.apply(field_data, domain, box));
    REQUIRE_NOTHROW(bc.apply(field_data, domain, box));
    REQUIRE_NOTHROW(bc.apply(field_data, domain, box));
  }
}

TEST_CASE("app-local FixedBC/MovingBC JSON catalog", "[boundary_conditions][unit]") {
  using pfc::ui::json;
  auto catalog = pfc::ui::make_builtin_field_modifier_catalog();
  pfc::ui::register_field_modifier<FixedBC>("fixed", catalog);
  pfc::ui::register_field_modifier<MovingBC>("moving", catalog);

  const json fixed_j = {
      {"type", "fixed"},
      {"rho_low", -0.5},
      {"rho_high", 0.5},
  };
  auto fixed = catalog.create_modifier("fixed", fixed_j);
  REQUIRE(fixed);
  REQUIRE(fixed->get_modifier_name() == "FixedBC");

  const json moving_j = {
      {"type", "moving"}, {"rho_low", -0.4}, {"rho_high", 0.2}, {"width", 20.0},
      {"alpha", 1.0},     {"disp", 0.0},     {"xpos", 10.0},
  };
  auto moving = catalog.create_modifier("moving", moving_j);
  REQUIRE(moving);
  REQUIRE(moving->get_modifier_name() == "MovingBC");
  REQUIRE(dynamic_cast<MovingBC *>(moving.get()) != nullptr);
  REQUIRE(dynamic_cast<MovingBC *>(moving.get())->get_xpos() == 10.0);
  // Without `mode`/`threshold` the tracker is amplitude at half the strongest
  // column.
  const auto *bc = dynamic_cast<MovingBC *>(moving.get());
  REQUIRE(bc->get_tracker() == tungsten::FrontTracker::amplitude);
  REQUIRE_FALSE(bc->has_explicit_threshold());
  REQUIRE(bc->get_threshold() == Catch::Approx(0.5));

  json mean_j = moving_j;
  mean_j["mode"] = "mean_field";
  auto mean = catalog.create_modifier("moving", mean_j);
  REQUIRE(dynamic_cast<MovingBC *>(mean.get())->get_tracker() ==
          tungsten::FrontTracker::mean_field);
  REQUIRE(dynamic_cast<MovingBC *>(mean.get())->get_threshold() ==
          Catch::Approx(-0.1));

  json amplitude_j = moving_j;
  amplitude_j["mode"] = "amplitude";
  amplitude_j["threshold"] = 0.3;
  auto amplitude = catalog.create_modifier("moving", amplitude_j);
  REQUIRE(dynamic_cast<MovingBC *>(amplitude.get())->get_tracker() ==
          tungsten::FrontTracker::amplitude);
  REQUIRE(dynamic_cast<MovingBC *>(amplitude.get())->get_threshold() == 0.3);

  json cutoff_j = moving_j;
  cutoff_j["mode"] = "cutoff";
  auto cutoff = catalog.create_modifier("moving", cutoff_j);
  REQUIRE(dynamic_cast<MovingBC *>(cutoff.get())->get_threshold() == 0.1);

  json bad_mode = moving_j;
  bad_mode["mode"] = "peak";
  REQUIRE_THROWS_AS(catalog.create_modifier("moving", bad_mode),
                    std::invalid_argument);
  json bad_threshold = moving_j;
  bad_threshold["threshold"] = "0.2";
  REQUIRE_THROWS_AS(catalog.create_modifier("moving", bad_threshold),
                    std::invalid_argument);
}
