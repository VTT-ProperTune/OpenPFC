// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <iostream>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "openpfc/boundary_conditions/moving_bc.hpp"
#include "openpfc/core/decomposition.hpp"
#include "openpfc/core/types.hpp"
#include "openpfc/core/world.hpp"
#include "openpfc/factory/decomposition_factory.hpp"
#include "openpfc/fft.hpp"
#include "openpfc/model.hpp"

using namespace pfc;
using Catch::Approx;
using pfc::types::Int3;

// Mock model class for testing
class ModelWithMovingBC : public Model {
public:
  ModelWithMovingBC(FFT &fft, const pfc::World &world) : pfc::Model(fft, world) {}

  void step(double /*t*/) override {}
  void initialize(double /*dt*/) override {}
};

TEST_CASE("MovingBC - Parameter Access", "[bc_moving]") {
  MovingBC bc;

  SECTION("Set and get rho_low") {
    bc.set_rho_low(0.1);
    // No getter for rho_low, but set should not throw
  }

  SECTION("Set and get rho_high") {
    bc.set_rho_high(0.9);
    // No getter for rho_high, but set should not throw
  }

  SECTION("Set and get xpos") {
    bc.set_xpos(50.0);
    REQUIRE(bc.get_xpos() == Approx(50.0));
  }

  SECTION("Set and get xwidth") {
    bc.set_xwidth(20.0);
    REQUIRE(bc.get_xwidth() == Approx(20.0));
  }

  SECTION("Set and get alpha") {
    bc.set_alpha(2.0);
    // No getter for alpha
  }

  SECTION("Set and get threshold") {
    bc.set_threshold(0.2);
    REQUIRE(bc.get_threshold() == Approx(0.2));
  }

  SECTION("Set and get disp") {
    bc.set_disp(30.0);
    // No getter for disp
  }
}

TEST_CASE("MovingBC - Constructor with Parameters", "[bc_moving]") {
  MovingBC bc(0.2, 0.8);
  // Constructor sets rho_low and rho_high
  // Can't verify directly without getters, but should not throw
}

TEST_CASE("MovingBC - Modifier Name", "[bc_moving]") {
  MovingBC bc;
  REQUIRE(bc.get_modifier_name() == "MovingBC");
}

TEST_CASE("MovingBC - Field Application", "[bc_moving]") {
  auto world =
      world::create(GridSize({32, 8, 8}), PhysicalOrigin({-128.0, -32.0, -32.0}),
                    GridSpacing({8.0, 8.0, 8.0}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  ModelWithMovingBC m(fft, world);

  const size_t field_size = fft.size_inbox();
  std::vector<double> psi(field_size, 0.0);
  m.add_real_field("default", psi);

  MovingBC bc(0.0, 1.0);
  bc.set_xwidth(15.0);
  bc.set_xpos(0.0);
  bc.set_threshold(0.1);

  SECTION("Apply boundary condition") {
    REQUIRE_NOTHROW(bc.apply(m, 0.0));

    const Field &field = m.get_real_field("default");

    // Check that field has been modified
    bool has_nonzero = false;
    for (const auto &value : field) {
      if (value != 0.0) {
        has_nonzero = true;
        break;
      }
    }
    // MovingBC modifies field in boundary region
    REQUIRE(has_nonzero);
  }

  SECTION("Field values in range") {
    bc.apply(m, 0.0);
    const Field &field = m.get_real_field("default");

    // Values should be between rho_low and rho_high
    for (const auto &value : field) {
      REQUIRE(value >= -0.1); // rho_low with small tolerance
      REQUIRE(value <= 1.1);  // rho_high with small tolerance
    }
  }

  SECTION("Multiple applications") {
    // First application
    bc.apply(m, 0.0);
    double xpos1 = bc.get_xpos();

    // Fill field with values that would trigger movement
    Field &field = m.get_real_field("default");
    std::fill(field.begin(), field.end(), 0.5); // Above threshold

    // Second application - boundary should move
    bc.apply(m, 0.0);
    double xpos2 = bc.get_xpos();

    // Position should have moved (or stayed same if no interface detected)
    REQUIRE(xpos2 >= xpos1);
  }
}

TEST_CASE("MovingBC - Integration with Model", "[bc_moving]") {
  auto world = world::create(GridSize({16, 8, 8}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  ModelWithMovingBC model(fft, world);

  const size_t field_size = fft.size_inbox();
  std::vector<double> psi(field_size, 0.0);
  model.add_real_field("density", psi);

  MovingBC bc;
  bc.set_field_name("density");
  bc.set_rho_low(0.1);
  bc.set_rho_high(0.9);
  bc.set_xwidth(10.0);

  REQUIRE_NOTHROW(bc.apply(model, 0.0));

  const Field &field = model.get_real_field("density");
  REQUIRE(field.size() == field_size);
}

TEST_CASE("MovingBC - Field Name Assignment", "[bc_moving]") {
  MovingBC bc;
  bc.set_field_name("interface");
  REQUIRE(bc.get_field_name() == "interface");
}

TEST_CASE("MovingBC - Boundary Position Tracking", "[bc_moving]") {
  auto world =
      world::create(GridSize({32, 8, 8}), PhysicalOrigin({-128.0, -32.0, -32.0}),
                    GridSpacing({8.0, 8.0, 8.0}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  ModelWithMovingBC m(fft, world);

  const size_t field_size = fft.size_inbox();
  std::vector<double> psi(field_size, 0.0);
  m.add_real_field("default", psi);

  MovingBC bc(0.0, 1.0);
  bc.set_xpos(100.0);
  bc.set_xwidth(15.0);

  SECTION("Position persists") {
    REQUIRE(bc.get_xpos() == Approx(100.0));
    bc.apply(m, 0.0);
    // Position may change after apply, but should remain valid
    REQUIRE(bc.get_xpos() >= 100.0); // Monotonic movement expected
  }
}
