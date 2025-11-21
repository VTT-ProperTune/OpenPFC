// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/boundary_conditions/fixed_bc.hpp"
#include "openpfc/core/world.hpp"
#include "openpfc/model.hpp"
#include <catch2/catch_test_macros.hpp>

#include "fixtures/mock_model.hpp"

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

  // TODO: Test actual apply() behavior requires integration test with real Model
  // that implements get_real_field(). See user story for FieldModifier redesign.
}
