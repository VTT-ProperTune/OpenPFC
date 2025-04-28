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
  const World world({128, 128, 128});

  SECTION("Construction and getters") {
    Decomposition decomposition = make_decomposition(world, 0, 1);

    REQUIRE(decomposition.get_world() == world);
  }

  /*
  SECTION("Domain decomposition status") {
    Decomposition decomposition = make_decomposition(world, 0, 1);

    std::ostringstream oss;
    oss << decomposition;

    std::string expectedOutput = R"EXPECTED(***** DOMAIN DECOMPOSITION STATUS *****
Real-to-complex symmetry is used (r2c direction = x)
Domain is split into 1 parts (minimum surface processor grid: [1, 1, 1])
Domain in real space: [128, 128, 128] (2097152 indexes)
Domain in complex space: [65, 128, 128] (1064960 indexes)
Domain 1/1: [0, 0, 0] x [127, 127, 127] (2097152 indexes) => [0, 0, 0] x [64, 127, 127] (1064960 indexes)
)EXPECTED";

    REQUIRE(oss.str() == expectedOutput);
  }
  */
}
