// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <stdexcept>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <mpi.h>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/ui/from_json_log.hpp>
#include <openpfc/frontend/ui/simulation_wiring_writers.hpp>
#include <openpfc/frontend/ui/spectral_json_driver_hooks.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/simulation/simulator.hpp>
#include <openpfc/kernel/simulation/time.hpp>

#include <fixtures/mock_model.hpp>

TEST_CASE("configure_spectral_json_driver_hooks sets from_json log rank",
          "[ui][hooks]") {
  pfc::ui::configure_spectral_json_driver_hooks(MPI_COMM_WORLD, 41);
  REQUIRE(pfc::ui::get_from_json_log_rank() == 41);
  pfc::ui::configure_spectral_json_driver_hooks(MPI_COMM_WORLD, -1);
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
