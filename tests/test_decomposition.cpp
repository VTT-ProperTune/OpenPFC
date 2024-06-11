/*

OpenPFC, a simulation software for the phase field crystal method.
Copyright (C) 2024 VTT Technical Research Centre of Finland Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.

*/

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <openpfc/decomposition.hpp>

using namespace Catch::Matchers;
using namespace pfc;

TEST_CASE("Decomposition class tests", "[Decomposition]") {
  // Create a dummy World object for testing
  const World world({128, 128, 128});

  SECTION("Construction and getters") {
    Decomposition decomposition(world, 0, 1);

    REQUIRE(decomposition.get_world() == world);
    REQUIRE(decomposition.get_rank() == 0);
  }

  SECTION("Domain decomposition status") {
    Decomposition decomposition(world, 0, 1);

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
}
