// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/core/decomposition.hpp"
#include "openpfc/factory/decomposition_factory.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace Catch::Matchers;
using namespace pfc;

TEST_CASE("Decomposition class tests", "[Decomposition]") {
  // Create a dummy World object for testing
  const World world = create_world({128, 128, 128});

  SECTION("Construction and getters") {
    Decomposition decomposition = make_decomposition(world, 0, 1);

    REQUIRE(decomposition.get_world() == world);
  }
}
