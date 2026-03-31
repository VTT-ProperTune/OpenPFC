// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>

using namespace Catch::Matchers;
using namespace pfc;
using namespace pfc::types;

TEST_CASE("Decomposition - basic functionality", "[decomposition][unit]") {
  // Create a dummy World object for testing
  auto world = world::create(GridSize({128, 128, 128}));

  SECTION("Construction and getters") {
    auto decomposition = decomposition::create(world, 1);

    REQUIRE(get_world(decomposition) == world);
  }

  SECTION("make_decomposition(world, rank, num_domains) matches domain count") {
    auto decomposition = make_decomposition(world, 0, 8);

    REQUIRE(decomposition::get_num_domains(decomposition) == 8);
    REQUIRE(decomposition::get_world(decomposition) == world);
  }
}
