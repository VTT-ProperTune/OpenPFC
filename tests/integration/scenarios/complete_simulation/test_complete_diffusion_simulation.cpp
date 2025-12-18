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

TEST_CASE("Complete diffusion simulation pipeline",
          "[integration][complete][diffusion]") {
  // Small end-to-end run
  auto world = world::uniform(32, 1.0);
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  auto decomp = decomposition::create(world, size);
  auto fft = fft::create(decomp);

  DiffusionModel model(fft, world);
  model.initialize(1.0e-3); // dt encoded in operator

  // Compute initial diagnostics
  double sum0 = 0.0, l2_0 = 0.0;
  for (const auto &v : model.m_psi) {
    sum0 += v;
    l2_0 += v * v;
  }

  // Run 50 steps (time parameter unused in linear diffusion)
  const int steps = 50;
  for (int s = 0; s < steps; ++s) {
    model.step(0.0);
  }

  // Basic validations: bounded values, finite norms
  double sum1 = 0.0, l2_1 = 0.0;
  for (const auto &v : model.m_psi) {
    sum1 += v;
    l2_1 += v * v;
  }
  REQUIRE(std::isfinite(l2_1));
  REQUIRE(l2_1 > 0.0);

  // Diffusion conserves mass (DC mode unchanged) and decreases L2 norm
  const double N = static_cast<double>(model.m_psi.size());
  REQUIRE(sum1 / N == Approx(sum0 / N).margin(1e-12));
  REQUIRE(l2_1 <= l2_0 + 1e-12);
}
