// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <sstream> // Include this for std::ostringstream

#include "openpfc/core/world.hpp"

using namespace Catch::Matchers;
using namespace pfc;

TEST_CASE("World - Construction and Accessors", "[world]") {
  SECTION("Construct World with valid dimensions, origin, and spacing") {
    Int3 dimensions = {100, 200, 300};
    Real3 origin = {1.0, 2.0, 3.0};
    Real3 spacing = {0.1, 0.2, 0.3};

    World world = world::create(dimensions, origin, spacing);

    REQUIRE(get_size(world) == dimensions);
    REQUIRE(get_origin(world) == origin);
    REQUIRE(get_spacing(world) == spacing);
  }

  SECTION("Construct World with default origin and spacing") {
    Int3 dimensions = {100, 200, 300};
    World world = world::create(dimensions);

    REQUIRE(get_size(world) == dimensions);
    REQUIRE(get_origin(world) == Real3{0.0, 0.0, 0.0});
    REQUIRE(get_spacing(world) == Real3{1.0, 1.0, 1.0});
  }

  SECTION("Invalid dimensions throw exception") {
    REQUIRE_THROWS_AS(world::create({-1, 100, 100}), std::invalid_argument);
    REQUIRE_THROWS_AS(world::create({0, 100, 100}), std::invalid_argument);
  }

  SECTION("Invalid spacing throws exception") {
    REQUIRE_THROWS_AS(
        world::create({100, 100, 100}, {0.0, 0.0, 0.0}, {-1.0, 1.0, 1.0}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        world::create({100, 100, 100}, {0.0, 0.0, 0.0}, {0.0, 1.0, 1.0}),
        std::invalid_argument);
  }
}

TEST_CASE("World - Coordinate Transformations", "[world]") {
  Int3 dimensions = {100, 100, 100};
  Real3 origin = {0.0, 0.0, 0.0};
  Real3 spacing = {1.0, 1.0, 1.0};
  World world = world::create(dimensions, origin, spacing);

  SECTION("Physical coordinates from grid indices") {
    Int3 indices = {10, 20, 30};
    Real3 expected_coords = {10.0, 20.0, 30.0};
    REQUIRE(to_coords(world, indices) == expected_coords);
  }

  SECTION("Grid indices from physical coordinates") {
    Real3 coords = {10.0, 20.0, 30.0};
    Int3 expected_indices = {10, 20, 30};
    REQUIRE(to_indices(world, coords) == expected_indices);
  }

  SECTION("Out-of-bounds physical coordinates") {
    Real3 coords = {-1.0, -1.0, -1.0};
    Int3 indices = to_indices(world, coords);
    REQUIRE(indices[0] < 0);
    REQUIRE(indices[1] < 0);
    REQUIRE(indices[2] < 0);
  }

  SECTION("Non-integer grid indices") {
    Real3 coords = {10.5, 20.5, 30.5};
    Int3 indices = to_indices(world, coords);
    REQUIRE(indices == Int3{10, 20, 30});
  }
}

TEST_CASE("World - Total Size", "[world]") {
  SECTION("Correct total size calculation") {
    Int3 dimensions = {10, 20, 30};
    World world = world::create(dimensions);
    REQUIRE(total_size(world) == 10 * 20 * 30);
  }
}

TEST_CASE("World - Equality and Inequality Operators", "[world]") {
  Int3 dimensions = {100, 100, 100};
  Real3 origin = {0.0, 0.0, 0.0};
  Real3 spacing = {1.0, 1.0, 1.0};

  World world1 = world::create(dimensions, origin, spacing);
  World world2 = world::create(dimensions, origin, spacing);
  World world3 = world::create({200, 100, 100}, origin, spacing);

  SECTION("Equality operator") {
    REQUIRE(world1 == world2);
    REQUIRE_FALSE(world1 == world3);
  }

  SECTION("Inequality operator") {
    REQUIRE(world1 != world3);
    REQUIRE_FALSE(world1 != world2);
  }
}
