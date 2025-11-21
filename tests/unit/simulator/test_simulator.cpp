// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <memory>

#include <catch2/catch_test_macros.hpp>

#include "openpfc/core/world.hpp"
#include "openpfc/factory/decomposition_factory.hpp"
#include "openpfc/model.hpp"
#include "openpfc/simulator.hpp"

#include "fixtures/mock_model.hpp"

using namespace pfc;
using namespace pfc::types;

TEST_CASE("Simulator functionality", "[simulator][unit]") {
  auto world = world::create({8, 8, 8});
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);

  pfc::testing::MockModel model(world);
  model.set_fft(fft);

  SECTION("Add and apply initial conditions") {
    Time time({0.0, 10.0, 1.0}, 1.0);
    Simulator simulator(model, time);

    // Add a field called "phi" to the model and fill it with zeros
    size_t N = 1;
    std::vector<double> phi(N, 0.0);
    model.add_field("phi", phi);

    // Create a FieldModifier object
    pfc::testing::MockIC ic;
    ic.set_field_name("phi");

    // Create unique pointer to the FieldModifier object
    std::unique_ptr<FieldModifier> ptr_ic =
        std::make_unique<pfc::testing::MockIC>(ic);

    // Add initial condition to the simulator
    simulator.add_initial_conditions(std::move(ptr_ic));
    REQUIRE(simulator.get_initial_conditions().size() == 1);

    // Apply initial conditions to the model
    REQUIRE(model.get_real_field("phi")[0] == 0.0);
    simulator.apply_initial_conditions();
    REQUIRE(model.get_real_field("phi")[0] == 1.0);
  }

  SECTION("Add and apply boundary conditions") {
    Time time({0.0, 10.0, 1.0}, 1.0);
    Simulator simulator(model, time);

    // Add a field called "phi" to the model and fill it with zeros
    size_t N = 1;
    std::vector<double> phi(N, 0.0);
    model.add_field("phi", phi);

    // Create a FieldModifier object
    pfc::testing::MockIC bc;
    bc.set_field_name("phi");

    // Create unique pointer to the FieldModifier object
    std::unique_ptr<FieldModifier> ptr_bc =
        std::make_unique<pfc::testing::MockIC>(bc);

    // Add boundary condition to the simulator
    simulator.add_boundary_conditions(std::move(ptr_bc));
    REQUIRE(simulator.get_boundary_conditions().size() == 1);

    // Apply boundary conditions to the model
    REQUIRE(model.get_real_field("phi")[0] == 0.0);
    simulator.apply_boundary_conditions();
    REQUIRE(model.get_real_field("phi")[0] == 1.0);
  }
}

TEST_CASE("Simulator - MockModel Integration", "[simulator]") {
  auto world = world::create({8, 8, 8});
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);

  pfc::testing::MockModel model(world);
  model.set_fft(fft);

  REQUIRE_NOTHROW(model.get_fft());
  REQUIRE(get_size(model.get_world()) == Int3{8, 8, 8});
}
