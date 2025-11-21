// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "openpfc/core/decomposition.hpp"
#include "openpfc/factory/decomposition_factory.hpp"

using namespace Catch::Matchers;
using namespace pfc;

TEST_CASE("Decomposition - basic functionality", "[decomposition][unit]") {
  // Create a dummy World object for testing
  auto world = world::create({128, 128, 128});

  SECTION("Construction and getters") {
    auto decomposition = decomposition::create(world, 1);

    REQUIRE(get_world(decomposition) == world);
  }
}
