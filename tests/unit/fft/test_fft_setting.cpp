// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>

#include <catch2/catch_test_macros.hpp>

#include "openpfc/core/decomposition.hpp"
#include "openpfc/core/world.hpp"
#include "openpfc/factory/decomposition_factory.hpp"
#include "openpfc/fft.hpp"
#include "openpfc/model.hpp"

#include "fixtures/mock_model.hpp"

using namespace pfc;

TEST_CASE("Model - FFT Setting and Retrieval", "[fft_setting]") {
  auto world = world::create(GridSize({8}), PhysicalOrigin({8}), GridSpacing({8}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);

  pfc::testing::MockModel model(world);

  model.set_fft(fft);

  // Ensure FFT object is set before proceeding
  REQUIRE_NOTHROW(model.get_fft());

  SECTION("Retrieve FFT object") {
    FFT &retrieved_fft = model.get_fft();

    // Ensure the retrieved FFT object matches the original
    REQUIRE(&retrieved_fft == &fft);
  }

  SECTION("Ensure FFT object is not null") {
    REQUIRE_NOTHROW(model.get_fft()); // Ensure no exception is thrown
  }
}
