// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "openpfc/kernel/data/world.hpp"
#include "openpfc/kernel/decomposition/decomposition.hpp"
#include "openpfc/kernel/decomposition/decomposition_factory.hpp"

using namespace pfc;

TEST_CASE("Decomposition - comprehensive (stub)",
          "[decomposition][comprehensive][unit]") {
  auto world = world::create(GridSize({128, 128, 128}));
  auto decomposition = decomposition::create(world, 1);
  REQUIRE(get_world(decomposition) == world);
}
