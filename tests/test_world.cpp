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
    World::Int3 dimensions = {100, 200, 300};
    World::Real3 origin = {1.0, 2.0, 3.0};
    World::Real3 spacing = {0.1, 0.2, 0.3};

    World world = create_world(dimensions, origin, spacing);

    REQUIRE(world.get_size() == dimensions);
    REQUIRE(world.get_origin() == origin);
    REQUIRE(world.get_spacing() == spacing);
  }

  SECTION("Construct World with default origin and spacing") {
    World::Int3 dimensions = {100, 200, 300};
    World world = create_world(dimensions);

    REQUIRE(world.get_size() == dimensions);
    REQUIRE(world.get_origin() == World::Real3{0.0, 0.0, 0.0});
    REQUIRE(world.get_spacing() == World::Real3{1.0, 1.0, 1.0});
  }

  SECTION("Invalid dimensions throw exception") {
    REQUIRE_THROWS_AS(create_world({-1, 100, 100}), std::invalid_argument);
    REQUIRE_THROWS_AS(create_world({0, 100, 100}), std::invalid_argument);
  }

  SECTION("Invalid spacing throws exception") {
    REQUIRE_THROWS_AS(create_world({100, 100, 100}, {0.0, 0.0, 0.0}, {-1.0, 1.0, 1.0}), std::invalid_argument);
    REQUIRE_THROWS_AS(create_world({100, 100, 100}, {0.0, 0.0, 0.0}, {0.0, 1.0, 1.0}), std::invalid_argument);
  }
}

TEST_CASE("World - Coordinate Transformations", "[world]") {
  World::Int3 dimensions = {100, 100, 100};
  World::Real3 origin = {0.0, 0.0, 0.0};
  World::Real3 spacing = {1.0, 1.0, 1.0};
  World world = create_world(dimensions, origin, spacing);

  SECTION("Physical coordinates from grid indices") {
    World::Int3 indices = {10, 20, 30};
    World::Real3 expected_coords = {10.0, 20.0, 30.0};
    REQUIRE(world.physical_coordinates(indices) == expected_coords);
  }

  SECTION("Grid indices from physical coordinates") {
    World::Real3 coords = {10.0, 20.0, 30.0};
    World::Int3 expected_indices = {10, 20, 30};
    REQUIRE(world.grid_indices(coords) == expected_indices);
  }

  SECTION("Out-of-bounds physical coordinates") {
    World::Real3 coords = {-1.0, -1.0, -1.0};
    World::Int3 indices = world.grid_indices(coords);
    REQUIRE(indices[0] < 0);
    REQUIRE(indices[1] < 0);
    REQUIRE(indices[2] < 0);
  }

  SECTION("Non-integer grid indices") {
    World::Real3 coords = {10.5, 20.5, 30.5};
    World::Int3 indices = world.grid_indices(coords);
    REQUIRE(indices == World::Int3{10, 20, 30});
  }
}

TEST_CASE("World - Total Size", "[world]") {
  SECTION("Correct total size calculation") {
    World::Int3 dimensions = {10, 20, 30};
    World world = create_world(dimensions);
    REQUIRE(world.total_size() == 10 * 20 * 30);
  }
}

TEST_CASE("World - Equality and Inequality Operators", "[world]") {
  World::Int3 dimensions = {100, 100, 100};
  World::Real3 origin = {0.0, 0.0, 0.0};
  World::Real3 spacing = {1.0, 1.0, 1.0};

  World world1 = create_world(dimensions, origin, spacing);
  World world2 = create_world(dimensions, origin, spacing);
  World world3 = create_world({200, 100, 100}, origin, spacing);

  SECTION("Equality operator") {
    REQUIRE(world1 == world2);
    REQUIRE_FALSE(world1 == world3);
  }

  SECTION("Inequality operator") {
    REQUIRE(world1 != world3);
    REQUIRE_FALSE(world1 != world2);
  }
}

TEST_CASE("World - Output Stream", "[world]") {
  World::Int3 dimensions = {100, 200, 300};
  World::Real3 origin = {1.0, 2.0, 3.0};
  World::Real3 spacing = {0.1, 0.2, 0.3};
  World world = create_world(dimensions, origin, spacing);

  std::ostringstream oss;
  oss << world;

  std::string expected_output = "(size = {100, 200, 300}, origin = {1.00, 2.00, 3.00}, spacing = {0.10, 0.20, 0.30})";
  REQUIRE(oss.str() == expected_output);
}
