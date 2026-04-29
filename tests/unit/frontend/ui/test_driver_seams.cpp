// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <openpfc/frontend/ui/from_json_log.hpp>
#include <openpfc/frontend/ui/spectral_json_driver_hooks.hpp>
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

TEST_CASE("write_scheduled_simulator_results bumps result counter",
          "[ui][simulator]") {
  auto world = pfc::world::create(pfc::GridSize({4, 4, 4}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = pfc::decomposition::create(world, 1);
  auto fft = pfc::fft::create(decomposition);
  pfc::testing::MockModel model(fft, world);
  pfc::Time time({0.0, 1.0, 0.1}, 1.0);
  pfc::Simulator sim(model, time, MPI_COMM_WORLD);

  REQUIRE(sim.get_result_counter() == 0);
  pfc::write_scheduled_simulator_results(sim);
  REQUIRE(sim.get_result_counter() == 1);
}
