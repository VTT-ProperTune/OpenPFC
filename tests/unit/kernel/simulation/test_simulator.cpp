// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <memory>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mpi.h>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/mpi/mpi.hpp>
#include <openpfc/kernel/simulation/model.hpp>
#include <openpfc/kernel/simulation/simulator.hpp>

#include "fixtures/mock_model.hpp"
#include "fixtures/simulation_factories.hpp"

using namespace Catch::Matchers;
using namespace pfc;
using namespace pfc::types;

TEST_CASE("Simulator functionality", "[simulator][unit]") {
  auto world = pfc::test::make_world_cube_8();
  auto decomposition = pfc::test::make_serial_decomposition(world);
  auto fft = fft::create(decomposition);

  pfc::testing::MockModel model(fft, world);

  SECTION("is_rank0 tracks simulator MPI communicator") {
    Time time({0.0, 10.0, 1.0}, 1.0);
    Simulator simulator(model, time, MPI_COMM_WORLD);
    REQUIRE(simulator.is_rank0() == (mpi::get_rank(MPI_COMM_WORLD) == 0));
    REQUIRE(pfc::is_rank0(simulator) == simulator.is_rank0());
    simulator.set_mpi_comm(MPI_COMM_WORLD);
    REQUIRE(simulator.is_rank0() == (mpi::get_rank(MPI_COMM_WORLD) == 0));
  }

  SECTION("results_writers exposes the registered writer map") {
    Time time({0.0, 10.0, 1.0}, 1.0);
    Simulator simulator(model, time);
    REQUIRE(pfc::results_writers(simulator).empty());
  }

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
  auto world = pfc::test::make_world_cube_8();
  auto decomposition = pfc::test::make_serial_decomposition(world);
  auto fft = fft::create(decomposition);

  pfc::testing::MockModel model(fft, world);

  REQUIRE_NOTHROW(get_fft(model));
  REQUIRE(get_size(get_world(model)) == Int3{8, 8, 8});
}

TEST_CASE("Simulator::step advances Time before Model::step", "[simulator][unit]") {
  auto world = pfc::test::make_world_cube_8();
  auto decomposition = pfc::test::make_serial_decomposition(world);
  auto fft = fft::create(decomposition);

  pfc::testing::InstrumentedMockModel model(fft, world);
  Time time({0.0, 1.5, 0.5}, 0.0);
  Simulator sim(model, time);
  pfc::initialize(sim);

  REQUIRE_FALSE(pfc::done(sim));
  pfc::step(sim);
  REQUIRE(model.step_call_count == 1);
  REQUIRE_THAT(model.last_step_time, WithinAbs(0.5, 1e-10));
  REQUIRE(pfc::get_increment(sim) == 1);
  REQUIRE_THAT(pfc::time::current(sim.get_time()), WithinAbs(0.5, 1e-10));

  pfc::step(sim);
  REQUIRE(model.step_call_count == 2);
  REQUIRE_THAT(model.last_step_time, WithinAbs(1.0, 1e-10));
  REQUIRE(pfc::get_increment(sim) == 2);

  pfc::step(sim);
  REQUIRE(model.step_call_count == 3);
  REQUIRE_THAT(model.last_step_time, WithinAbs(1.5, 1e-10));
  REQUIRE(pfc::get_increment(sim) == 3);
  REQUIRE(pfc::done(sim));
}

TEST_CASE("Simulator begin/end/step_with_physics matches step()",
          "[simulator][unit]") {
  auto world = pfc::test::make_world_cube_8();
  auto decomposition = pfc::test::make_serial_decomposition(world);
  auto fft = fft::create(decomposition);

  pfc::testing::InstrumentedMockModel model_a(fft, world);
  pfc::testing::InstrumentedMockModel model_b(fft, world);
  Time time_a({0.0, 1.5, 0.5}, 0.0);
  Time time_b({0.0, 1.5, 0.5}, 0.0);
  Simulator sim_a(model_a, time_a);
  Simulator sim_b(model_b, time_b);
  pfc::initialize(sim_a);
  pfc::initialize(sim_b);

  while (!pfc::done(sim_a)) {
    pfc::step(sim_a);
  }
  while (!pfc::done(sim_b)) {
    sim_b.step_with_physics(
        [&] { pfc::step(model_b, pfc::time::current(sim_b.get_time())); });
  }

  REQUIRE(model_a.step_call_count == model_b.step_call_count);
  REQUIRE(model_a.step_call_count == 3);
  REQUIRE_THAT(model_b.last_step_time, WithinAbs(1.5, 1e-10));
}

TEST_CASE("Simulator phased begin/end matches step()", "[simulator][unit]") {
  auto world = pfc::test::make_world_cube_8();
  auto decomposition = pfc::test::make_serial_decomposition(world);
  auto fft = fft::create(decomposition);

  pfc::testing::InstrumentedMockModel model_a(fft, world);
  pfc::testing::InstrumentedMockModel model_b(fft, world);
  Time time_a({0.0, 1.5, 0.5}, 0.0);
  Time time_b({0.0, 1.5, 0.5}, 0.0);
  Simulator sim_a(model_a, time_a);
  Simulator sim_b(model_b, time_b);
  pfc::initialize(sim_a);
  pfc::initialize(sim_b);

  while (!pfc::done(sim_a)) {
    pfc::step(sim_a);
  }

  while (!pfc::done(sim_b)) {
    pfc::begin_integrator_step(sim_b);
    pfc::step(model_b, pfc::time::current(sim_b.get_time()));
    pfc::end_integrator_step(sim_b);
  }

  REQUIRE(model_a.step_call_count == model_b.step_call_count);
  REQUIRE(model_b.step_call_count == 3);
}
