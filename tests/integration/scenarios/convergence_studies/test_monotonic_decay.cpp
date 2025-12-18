// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fixtures/diffusion_model.hpp>
#include <fixtures/simulation_runner.hpp>
#include <openpfc/core/decomposition.hpp>
#include <openpfc/core/world.hpp>
#include <openpfc/fft.hpp>

using namespace pfc;
using namespace pfc::test;

TEST_CASE("Monotonic decay of peak amplitude in diffusion",
          "[integration][convergence]") {
  auto world = world::uniform(32, 1.0);
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  auto decomp = decomposition::create(world, size);
  auto fft = fft::create(decomp);

  DiffusionModel model(fft, world);
  model.initialize(1.0e-3);

  // Initial max and mean
  double mean0 = compute_mean(model.m_psi);
  double max0 = compute_max(model.m_psi);

  // Advance and re-measure
  SimulationRunner runner(model);
  runner.run_steps(100);
  double mean1 = compute_mean(model.m_psi);
  double max1 = compute_max(model.m_psi);

  // Mean conserved, peak decays
  REQUIRE(mean1 == Catch::Approx(mean0).margin(1e-12));
  REQUIRE(max1 <= max0 + 1e-12);
}
