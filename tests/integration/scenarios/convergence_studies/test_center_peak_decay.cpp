// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <fixtures/diffusion_model.hpp>
#include <openpfc/core/decomposition.hpp>
#include <openpfc/core/world.hpp>
#include <openpfc/fft.hpp>

using namespace pfc;
using namespace pfc::test;

TEST_CASE("Center peak amplitude decays under diffusion",
          "[integration][convergence]") {
  auto world = world::uniform(32, 1.0);
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  auto decomp = decomposition::create(world, size);
  auto fft = fft::create(decomp);

  DiffusionModel model(fft, world);
  model.initialize(1.0e-3);

  const int idx = model.get_midpoint_idx();
  if (idx < 0) {
    SUCCEED("Center point not on this rank - skip local check");
    return;
  }

  const double a0 = model.m_psi[idx];
  for (int i = 0; i < 200; ++i) model.step(0.0);
  const double a1 = model.m_psi[idx];

  REQUIRE(a1 <= a0 + 1e-12);
}
