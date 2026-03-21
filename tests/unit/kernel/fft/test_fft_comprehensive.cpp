// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/fft/fft.hpp>

using namespace pfc;

TEST_CASE("FFT - comprehensive (stub)", "[fft][comprehensive][unit]") {
  auto world = world::create(GridSize({8, 1, 1}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  REQUIRE(fft.size_inbox() > 0);
  REQUIRE(fft.size_outbox() > 0);
}
