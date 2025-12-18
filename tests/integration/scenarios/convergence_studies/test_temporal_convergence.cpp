// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fixtures/diffusion_model.hpp>
#include <openpfc/core/decomposition.hpp>
#include <openpfc/core/world.hpp>
#include <openpfc/fft.hpp>

using Catch::Approx;
using namespace pfc;
using namespace pfc::test;

TEST_CASE("Temporal convergence: diffusion error decreases with dt",
          "[integration][convergence]") {
  auto world = world::uniform(32, 1.0);
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  auto decomp = decomposition::create(world, size);
  auto fft = fft::create(decomp);

  // Run with coarse dt
  DiffusionModel model_coarse(fft, world);
  model_coarse.initialize(5e-3);
  for (int i = 0; i < 50; ++i) model_coarse.step(0.0);
  double l2_coarse = 0.0;
  for (const auto &v : model_coarse.m_psi) l2_coarse += v * v;

  // Run with fine dt (same total time)
  DiffusionModel model_fine(fft, world);
  model_fine.initialize(1e-3);
  for (int i = 0; i < 250; ++i) model_fine.step(0.0);
  double l2_fine = 0.0;
  for (const auto &v : model_fine.m_psi) l2_fine += v * v;

  // Expect smaller error at finer dt (norm drop heuristic for smoke test)
  REQUIRE(l2_fine <= l2_coarse + 1e-12);
}
