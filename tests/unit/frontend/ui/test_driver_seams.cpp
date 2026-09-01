// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <mpi.h>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/ui/from_json_log.hpp>
#include <openpfc/frontend/ui/json_driver_hooks.hpp>
#include <openpfc/frontend/ui/simulation_wiring_conditions.hpp>
#include <openpfc/frontend/ui/simulation_wiring_simulator_section.hpp>
#include <openpfc/frontend/ui/simulation_wiring_writers.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/simulation/simulator.hpp>
#include <openpfc/kernel/simulation/time.hpp>

#include <fixtures/mock_model.hpp>

TEST_CASE("configure_json_driver_hooks sets from_json log rank", "[ui][hooks]") {
  pfc::ui::configure_json_driver_hooks(MPI_COMM_WORLD, 41);
  REQUIRE(pfc::ui::get_from_json_log_rank() == 41);
  pfc::ui::configure_json_driver_hooks(MPI_COMM_WORLD, -1);
  REQUIRE(pfc::ui::get_from_json_log_rank() == -1);
}

TEST_CASE("pfc::write_results bumps result counter", "[ui][simulator]") {
  auto domain = pfc::domain::create(pfc::GridSize({4, 4, 4}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = pfc::decomposition::create(domain, 1);
  const auto &world = pfc::decomposition::get_world(decomposition);
  auto fft = pfc::fft::create(decomposition);
  pfc::testing::MockModel model(fft, world);
  pfc::Time time({0.0, 1.0, 0.1}, 1.0);
  pfc::Simulator sim(model, time, MPI_COMM_WORLD);

  REQUIRE(pfc::get_result_counter(sim) == 0);
  pfc::write_results(sim);
  REQUIRE(pfc::get_result_counter(sim) == 1);
}

TEST_CASE("unknown results writer type is a hard error", "[ui][writers]") {
  auto domain = pfc::domain::create(pfc::GridSize({4, 4, 4}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = pfc::decomposition::create(domain, 1);
  const auto &world = pfc::decomposition::get_world(decomposition);
  auto fft = pfc::fft::create(decomposition);
  pfc::testing::MockModel model(fft, world);
  pfc::Time time({0.0, 1.0, 0.1}, 1.0);
  pfc::Simulator sim(model, time, MPI_COMM_WORLD);
  const nlohmann::json settings = {
      {"saveat", 1.0},
      {"fields", {{{"name", "psi"}, {"data", "out.bin"}, {"writer", "nope"}}}}};
  const pfc::ui::JsonWiringContext ctx{MPI_COMM_WORLD, 0, true};
  REQUIRE_THROWS_AS(
      pfc::ui::add_result_writers_from_json(
          sim, settings, ctx, pfc::ui::default_results_writer_catalog()),
      std::invalid_argument);
  try {
    pfc::ui::add_result_writers_from_json(sim, settings, ctx,
                                          pfc::ui::default_results_writer_catalog());
  } catch (const std::invalid_argument &e) {
    REQUIRE_THAT(std::string(e.what()),
                 Catch::Matchers::ContainsSubstring("writer"));
    REQUIRE_THAT(std::string(e.what()),
                 Catch::Matchers::ContainsSubstring("binary"));
  }
}

TEST_CASE("missing initial-condition type is a hard error", "[ui][modifiers]") {
  auto domain = pfc::domain::create(pfc::GridSize({4, 4, 4}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = pfc::decomposition::create(domain, 1);
  const auto &world = pfc::decomposition::get_world(decomposition);
  auto fft = pfc::fft::create(decomposition);
  pfc::testing::MockModel model(fft, world);
  pfc::Time time({0.0, 1.0, 0.1}, 1.0);
  pfc::Simulator sim(model, time, MPI_COMM_WORLD);
  const nlohmann::json settings = {
      {"initial_conditions", {nlohmann::json::object()}}};
  const pfc::ui::JsonWiringContext ctx{MPI_COMM_WORLD, 0, true};
  REQUIRE_THROWS_AS(
      pfc::ui::add_initial_conditions_from_json(
          sim, settings, ctx, pfc::ui::default_field_modifier_catalog()),
      std::invalid_argument);
}

TEST_CASE("apply_simulator_section_from_json overlays integrator method",
          "[ui][integrator]") {
  auto domain = pfc::domain::create(pfc::GridSize({4, 4, 4}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = pfc::decomposition::create(domain, 1);
  const auto &world = pfc::decomposition::get_world(decomposition);
  auto fft = pfc::fft::create(decomposition);
  pfc::testing::MockModel model(fft, world);
  pfc::Time time({0.0, 1.0, 0.1}, 1.0);
  pfc::Simulator sim(model, time, MPI_COMM_WORLD);
  const nlohmann::json settings = {
      {"simulator", {{"integrator", {{"method", "rk4_classical"}}}}}};
  pfc::ui::apply_simulator_section_from_json(sim, time, settings);
  REQUIRE(time.method() == pfc::sim::steppers::RKIntegratorMethod::RK4_Classical);

  const nlohmann::json etd = {{"simulator", {{"integrator", {{"method", "etd1"}}}}}};
  pfc::ui::apply_simulator_section_from_json(sim, time, etd);
  REQUIRE(time.method() == pfc::sim::steppers::RKIntegratorMethod::ETD1);

  const nlohmann::json imex = {
      {"simulator", {{"integrator", {{"method", "imex_euler"}}}}}};
  pfc::ui::apply_simulator_section_from_json(sim, time, imex);
  REQUIRE(time.method() == pfc::sim::steppers::RKIntegratorMethod::ImexEuler);

  const nlohmann::json bad = {{"simulator", {{"integrator", {{"method", "imex"}}}}}};
  REQUIRE_THROWS_AS(pfc::ui::apply_simulator_section_from_json(sim, time, bad),
                    std::invalid_argument);

  const nlohmann::json mixed = {{"restart_from", "/tmp/ckpt/step_1"},
                                {"simulator", {{"increment", 4}}}};
  REQUIRE_THROWS_AS(pfc::ui::apply_simulator_section_from_json(sim, time, mixed),
                    std::invalid_argument);
}

TEST_CASE("JSON writer vtk writes a VTI file", "[ui][writers][vtk]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) {
    SKIP("serial VTI check requires one MPI rank");
  }

  auto domain = pfc::domain::create(pfc::GridSize({4, 4, 4}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = pfc::decomposition::create(domain, 1);
  const auto &world = pfc::decomposition::get_world(decomposition);
  auto fft = pfc::fft::create(decomposition);
  pfc::testing::MockModel model(fft, world);
  const auto inbox = pfc::fft::get_inbox(fft);
  const std::size_t n = static_cast<std::size_t>(inbox.size[0]) *
                        static_cast<std::size_t>(inbox.size[1]) *
                        static_cast<std::size_t>(inbox.size[2]);
  std::vector<double> psi(n, 0.0);
  pfc::add_field(model, "psi", psi);

  pfc::Time time({0.0, 1.0, 0.1}, 1.0);
  pfc::Simulator sim(model, time, MPI_COMM_WORLD);
  const nlohmann::json settings = {{"saveat", 1.0},
                                   {"fields",
                                    {{{"name", "psi"},
                                      {"data", "results/openpfc_json_vtk_%04d.vti"},
                                      {"writer", "vtk"}}}}};
  const pfc::ui::JsonWiringContext ctx{MPI_COMM_WORLD, 0, true};
  pfc::ui::add_result_writers_from_json(sim, settings, ctx,
                                        pfc::ui::default_results_writer_catalog());
  pfc::write_results(sim);
  REQUIRE(std::filesystem::exists("results/openpfc_json_vtk_0000.vti"));
}
