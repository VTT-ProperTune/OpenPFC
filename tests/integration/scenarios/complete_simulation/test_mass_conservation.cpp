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

TEST_CASE("Diffusion mass conservation",
          "[integration][complete][diffusion][mass]") {
  auto world = world::uniform(32, 1.0);
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  auto decomp = decomposition::create(world, size);
  auto fft = fft::create(decomp);

  DiffusionModel model(fft, world);
  model.initialize(1.0e-3);

  // Initial diagnostics
  double sum0 = 0.0, l2_0 = 0.0;
  for (const auto &v : model.m_psi) {
    sum0 += v;
    l2_0 += v * v;
  }

  // Advance
  for (int s = 0; s < 100; ++s) model.step(0.0);

  double sum1 = 0.0, l2_1 = 0.0;
  for (const auto &v : model.m_psi) {
    sum1 += v;
    l2_1 += v * v;
  }

  const double N = static_cast<double>(model.m_psi.size());
  REQUIRE(sum1 / N == Approx(sum0 / N).margin(1e-12));
  REQUIRE(l2_1 <= l2_0 + 1e-12);
}
