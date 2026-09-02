// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc_apps/fixed_bc.hpp>
#include <openpfc_apps/moving_bc.hpp>
#include <openpfc_apps/solidification_bc_json.hpp>

using namespace pfc;

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
    for (const auto &val : field_data)
      values_in_range &= val >= -2.6 && val <= 3.6;
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
}
