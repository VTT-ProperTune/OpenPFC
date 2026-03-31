// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <memory>

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/simulation/model.hpp>
#include <openpfc/kernel/simulation/simulator.hpp>

#include "fixtures/mock_model.hpp"

using namespace pfc;
using namespace pfc::types;

TEST_CASE("Simulator functionality", "[simulator][unit]") {
  auto world = world::create(GridSize({8, 8, 8}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);

  pfc::testing::MockModel model(fft, world);

  SECTION("Add and apply initial conditions") {
    Time time({0.0, 10.0, 1.0}, 1.0);
    Simulator simulator(model, time);

    // Add a field called "phi" to the model and fill it with zeros
    size_t N = 1;
    std::vector<double> phi(N, 0.0);
    add_field(model, "phi", phi);

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
    REQUIRE(get_real_field(model, "phi")[0] == 0.0);
    simulator.apply_initial_conditions();
    REQUIRE(get_real_field(model, "phi")[0] == 1.0);
  }

  SECTION("Multi-field initial condition registration") {
    Time time({0.0, 10.0, 1.0}, 1.0);
    Simulator simulator(model, time);

    std::vector<double> a(1, 0.0);
    std::vector<double> b(1, 0.0);
    add_field(model, "a", a);
    add_field(model, "b", b);

    auto ic = std::make_unique<pfc::testing::MockICMulti>();
    ic->set_field_names({"a", "b"});
    REQUIRE(simulator.add_initial_conditions(std::move(ic)));
    simulator.apply_initial_conditions();
    REQUIRE(get_real_field(model, "a")[0] == 2.0);
    REQUIRE(get_real_field(model, "b")[0] == 2.0);
  }

  SECTION("Add and apply boundary conditions") {
    Time time({0.0, 10.0, 1.0}, 1.0);
    Simulator simulator(model, time);

    // Add a field called "phi" to the model and fill it with zeros
    size_t N = 1;
    std::vector<double> phi(N, 0.0);
    add_field(model, "phi", phi);

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
    REQUIRE(get_real_field(model, "phi")[0] == 0.0);
    simulator.apply_boundary_conditions();
    REQUIRE(get_real_field(model, "phi")[0] == 1.0);
  }
}

TEST_CASE("Simulator - MockModel Integration", "[simulator]") {
  auto world = world::create(GridSize({8, 8, 8}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);

  pfc::testing::MockModel model(fft, world);

  REQUIRE_NOTHROW(get_fft(model));
  REQUIRE(get_size(get_world(model)) == Int3{8, 8, 8});
}
