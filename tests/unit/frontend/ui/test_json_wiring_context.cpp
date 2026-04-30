// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/ui/results_writer_catalog.hpp>
#include <openpfc/frontend/ui/simulation_wiring.hpp>
#include <openpfc/frontend/ui/simulation_wiring_context.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/simulation/time.hpp>

#include <fixtures/mock_model.hpp>

using json = nlohmann::json;

TEST_CASE("JsonWiringContext IC/BC wiring matches legacy overloads",
          "[ui][wiring]") {
  auto world = pfc::world::create(pfc::GridSize({4, 4, 4}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = pfc::decomposition::create(world, 1);
  auto fft = pfc::fft::create(decomposition);
  pfc::testing::MockModel model(fft, world);
  pfc::Time time({0.0, 1.0, 0.1}, 1.0);
  pfc::Simulator sim(model, time, MPI_COMM_WORLD);

  json settings = json::object();

  const pfc::ui::JsonWiringContext ctx{MPI_COMM_WORLD, 3, false};
  auto &mod_cat = pfc::ui::default_field_modifier_catalog();

  REQUIRE_NOTHROW(
      pfc::ui::add_initial_conditions_from_json(sim, settings, ctx, mod_cat));
  REQUIRE_NOTHROW(pfc::ui::add_initial_conditions_from_json(
      sim, settings, MPI_COMM_WORLD, 3, false, mod_cat));

  REQUIRE_NOTHROW(
      pfc::ui::add_boundary_conditions_from_json(sim, settings, ctx, mod_cat));
  REQUIRE_NOTHROW(pfc::ui::add_boundary_conditions_from_json(
      sim, settings, MPI_COMM_WORLD, 3, false, mod_cat));
}

TEST_CASE("wire_simulator_and_runtime_from_json accepts JsonWiringContext",
          "[ui][wiring]") {
  auto world = pfc::world::create(pfc::GridSize({4, 4, 4}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = pfc::decomposition::create(world, 1);
  auto fft = pfc::fft::create(decomposition);
  pfc::testing::MockModel model(fft, world);
  pfc::Time time({0.0, 1.0, 0.1}, 1.0);
  pfc::Simulator sim(model, time, MPI_COMM_WORLD);

  json settings = json::object();
  const pfc::ui::JsonWiringContext ctx{MPI_COMM_WORLD, 0, true};
  auto &mod_cat = pfc::ui::default_field_modifier_catalog();
  auto &res_cat = pfc::ui::default_results_writer_catalog();

  REQUIRE_NOTHROW(pfc::ui::wire_simulator_and_runtime_from_json(
      sim, time, settings, ctx, mod_cat, res_cat));
  REQUIRE_NOTHROW(pfc::ui::wire_simulator_and_runtime_from_json(
      sim, time, settings, MPI_COMM_WORLD, 0, true, mod_cat, res_cat));
}

TEST_CASE("wire_simulator_and_runtime_from_json accepts JsonWiringSession",
          "[ui][wiring]") {
  auto world = pfc::world::create(pfc::GridSize({4, 4, 4}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = pfc::decomposition::create(world, 1);
  auto fft = pfc::fft::create(decomposition);
  pfc::testing::MockModel model(fft, world);
  pfc::Time time({0.0, 1.0, 0.1}, 1.0);
  pfc::Simulator sim(model, time, MPI_COMM_WORLD);

  json settings = json::object();
  const pfc::ui::JsonWiringSession session = pfc::ui::make_json_wiring_session(
      MPI_COMM_WORLD, 0, true, pfc::ui::default_field_modifier_catalog(),
      pfc::ui::default_results_writer_catalog());

  REQUIRE_NOTHROW(
      pfc::ui::wire_simulator_and_runtime_from_json(sim, time, settings, session));
}
