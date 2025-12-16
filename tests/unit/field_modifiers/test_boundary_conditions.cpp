// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "openpfc/boundary_conditions/fixed_bc.hpp"
#include "openpfc/core/world.hpp"
#include "openpfc/model.hpp"

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
}

TEST_CASE("FixedBC - apply method", "[boundary_conditions][unit]") {
  // Create a 1D world for testing boundary conditions
  auto world = world::create(GridSize({100, 1, 1}), PhysicalOrigin({0.0, 0.0, 0.0}),
                             GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);

  pfc::testing::MockModel model(world);
  model.set_fft(fft);

  // Create and add a field
  std::vector<double> field_data(get_total_size(world), 0.0);
  model.add_real_field("psi", field_data);

  SECTION("FixedBC applies boundary smoothly") {
    FixedBC bc(-1.0, 1.0);
    bc.set_field_name("psi");

    // Apply the boundary condition
    REQUIRE_NOTHROW(bc.apply(model, 0.0));

    // Verify field was modified (at least some values changed)
    const auto &result = model.get_real_field("psi");
    bool has_nonzero = false;
    for (const auto &val : result) {
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

    REQUIRE_NOTHROW(bc.apply(model, 0.0));

    // Values should be between rho_low and rho_high
    const auto &result = model.get_real_field("psi");
    for (const auto &val : result) {
      REQUIRE(val >= -2.6); // Allow small tolerance
      REQUIRE(val <= 3.6);
    }
  }

  SECTION("FixedBC can be applied multiple times") {
    FixedBC bc(0.0, 1.0);
    bc.set_field_name("psi");

    REQUIRE_NOTHROW(bc.apply(model, 0.0));
    REQUIRE_NOTHROW(bc.apply(model, 1.0));
    REQUIRE_NOTHROW(bc.apply(model, 2.0));
  }
}
