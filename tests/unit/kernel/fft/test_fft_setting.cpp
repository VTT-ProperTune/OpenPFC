// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>

#include <catch2/catch_test_macros.hpp>

#include "openpfc/kernel/data/world.hpp"
#include "openpfc/kernel/decomposition/decomposition.hpp"
#include "openpfc/kernel/decomposition/decomposition_factory.hpp"
#include "openpfc/kernel/fft/fft.hpp"
#include "openpfc/kernel/simulation/model.hpp"

#include "fixtures/mock_model.hpp"

using namespace pfc;

TEST_CASE("Model - FFT Setting and Retrieval", "[fft_setting]") {
  auto world = world::create(GridSize({8, 8, 8}), PhysicalOrigin({8.0, 8.0, 8.0}),
                             GridSpacing({8.0, 8.0, 8.0}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);

  pfc::testing::MockModel model(fft, world);

  // Ensure FFT object is set before proceeding
  REQUIRE_NOTHROW(model.get_fft());

  SECTION("Retrieve FFT object") {
    FFT &retrieved_fft = model.get_fft();

    // Ensure the retrieved FFT object matches the original
    REQUIRE(&retrieved_fft == &fft);
  }

  SECTION("Ensure FFT object is always valid") { REQUIRE_NOTHROW(model.get_fft()); }
}
