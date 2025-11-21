// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <sstream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "openpfc/core/world.hpp"

using namespace Catch::Matchers;
using namespace pfc;
using namespace pfc::types;

TEST_CASE("World - construction and accessors", "[world][unit]") {
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

TEST_CASE("World - coordinate transformations", "[world][unit]") {
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

TEST_CASE("World - total size calculation", "[world][unit]") {
  SECTION("Correct total size calculation") {
    Int3 dimensions = {10, 20, 30};
    World world = world::create(dimensions);
    REQUIRE(get_total_size(world) == 10 * 20 * 30);
  }
}

TEST_CASE("World - equality and inequality operators", "[world][unit]") {
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

TEST_CASE("World - streaming output", "[world][unit]") {
  SECTION("World can be streamed to output") {
    Int3 dimensions = {10, 20, 30};
    Real3 origin = {1.0, 2.0, 3.0};
    Real3 spacing = {0.5, 0.5, 0.5};
    World world = world::create(dimensions, origin, spacing);

    std::ostringstream oss;
    REQUIRE_NOTHROW(oss << world);

    std::string output = oss.str();
    REQUIRE(output.find("World Summary") != std::string::npos);
    REQUIRE(output.find("Size") != std::string::npos);
    REQUIRE(output.find("10") != std::string::npos);
    REQUIRE(output.find("20") != std::string::npos);
    REQUIRE(output.find("30") != std::string::npos);
  }

  SECTION("World output includes coordinate system info") {
    World world = world::create({5, 5, 5}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
    std::ostringstream oss;
    oss << world;

    std::string output = oss.str();
    REQUIRE(output.find("Coordinate Sys") != std::string::npos);
    REQUIRE(output.find("Cartesian") != std::string::npos);
    REQUIRE(output.find("Offset") != std::string::npos);
    REQUIRE(output.find("Spacing") != std::string::npos);
  }

  SECTION("World output with non-default values") {
    Real3 origin = {1.5, 2.5, 3.5};
    Real3 spacing = {0.1, 0.2, 0.3};
    World world = world::create({8, 16, 32}, origin, spacing);

    std::ostringstream oss;
    oss << world;
    std::string output = oss.str();

    // Verify output contains key values (formatted with 2 decimal places)
    REQUIRE(output.find("1.50") != std::string::npos); // origin x
    REQUIRE(output.find("0.10") != std::string::npos); // spacing x
  }
}

TEST_CASE("World - uniform() helper creates correct grid",
          "[world][helpers][unit]") {
  SECTION("Single argument (unit spacing)") {
    auto world = world::uniform(64);

    REQUIRE(world::get_size(world) == Int3{64, 64, 64});
    REQUIRE(world::get_origin(world) == Real3{0.0, 0.0, 0.0});
    REQUIRE(world::get_spacing(world) == Real3{1.0, 1.0, 1.0});
  }

  SECTION("With custom spacing") {
    auto world = world::uniform(128, 0.5);

    REQUIRE(world::get_size(world) == Int3{128, 128, 128});
    REQUIRE(world::get_spacing(world) == Real3{0.5, 0.5, 0.5});
  }

  SECTION("Throws on invalid size") {
    REQUIRE_THROWS_AS(world::uniform(0), std::invalid_argument);
    REQUIRE_THROWS_AS(world::uniform(-10), std::invalid_argument);
  }

  SECTION("Throws on invalid spacing") {
    REQUIRE_THROWS_AS(world::uniform(64, 0.0), std::invalid_argument);
    REQUIRE_THROWS_AS(world::uniform(64, -0.1), std::invalid_argument);
  }
}

TEST_CASE("World - from_bounds() computes spacing correctly",
          "[world][helpers][unit]") {
  SECTION("Periodic grid") {
    auto world = world::from_bounds({100, 100, 100}, {0.0, 0.0, 0.0},
                                    {10.0, 10.0, 10.0}, {true, true, true});

    // Periodic: dx = (upper - lower) / size
    REQUIRE_THAT(world::get_spacing(world, 0), WithinAbs(0.1, 1e-10));
    REQUIRE_THAT(world::get_spacing(world, 1), WithinAbs(0.1, 1e-10));
    REQUIRE_THAT(world::get_spacing(world, 2), WithinAbs(0.1, 1e-10));
  }

  SECTION("Non-periodic grid") {
    auto world = world::from_bounds({100, 100, 100}, {0.0, 0.0, 0.0},
                                    {10.0, 10.0, 10.0}, {false, false, false});

    // Non-periodic: dx = (upper - lower) / (size - 1)
    double expected = 10.0 / 99.0;
    REQUIRE_THAT(world::get_spacing(world, 0), WithinAbs(expected, 1e-10));
  }

  SECTION("Mixed periodicity") {
    auto world =
        world::from_bounds({100, 100, 100}, {0.0, 0.0, 0.0}, {10.0, 20.0, 30.0},
                           {true, false, true} // Periodic in x and z, not y
        );

    REQUIRE_THAT(world::get_spacing(world, 0), WithinAbs(0.1, 1e-10)); // 10/100
    REQUIRE_THAT(world::get_spacing(world, 1),
                 WithinAbs(20.0 / 99.0, 1e-10));                       // 20/99
    REQUIRE_THAT(world::get_spacing(world, 2), WithinAbs(0.3, 1e-10)); // 30/100
  }

  SECTION("Validates inputs") {
    REQUIRE_THROWS_AS(world::from_bounds({0, 100, 100}, {0, 0, 0}, {10, 10, 10}),
                      std::invalid_argument);

    REQUIRE_THROWS_AS(world::from_bounds({100, 100, 100}, {0, 0, 0}, {0, 10, 10}),
                      std::invalid_argument);

    REQUIRE_THROWS_AS(world::from_bounds({-5, 100, 100}, {0, 0, 0}, {10, 10, 10}),
                      std::invalid_argument);
  }
}

TEST_CASE("World - with_spacing() helper", "[world][helpers][unit]") {
  SECTION("Custom spacing, default origin") {
    auto world = world::with_spacing({64, 64, 128}, {0.1, 0.1, 0.05});

    REQUIRE(world::get_size(world) == Int3{64, 64, 128});
    REQUIRE(world::get_origin(world) == Real3{0.0, 0.0, 0.0});
    REQUIRE(world::get_spacing(world) == Real3{0.1, 0.1, 0.05});
  }

  SECTION("Validates inputs") {
    REQUIRE_THROWS_AS(world::with_spacing({-1, 64, 64}, {0.1, 0.1, 0.1}),
                      std::invalid_argument);

    REQUIRE_THROWS_AS(world::with_spacing({64, 64, 64}, {0.0, 0.1, 0.1}),
                      std::invalid_argument);

    REQUIRE_THROWS_AS(world::with_spacing({64, 64, 64}, {-0.1, 0.1, 0.1}),
                      std::invalid_argument);
  }
}

TEST_CASE("World - with_origin() helper", "[world][helpers][unit]") {
  SECTION("Custom origin, unit spacing") {
    auto world = world::with_origin({64, 64, 64}, {-5.0, -5.0, 0.0});

    REQUIRE(world::get_size(world) == Int3{64, 64, 64});
    REQUIRE(world::get_origin(world) == Real3{-5.0, -5.0, 0.0});
    REQUIRE(world::get_spacing(world) == Real3{1.0, 1.0, 1.0});
  }

  SECTION("Validates size") {
    REQUIRE_THROWS_AS(world::with_origin({0, 64, 64}, {0.0, 0.0, 0.0}),
                      std::invalid_argument);
  }
}

TEST_CASE("World - Helpers produce same result as create()",
          "[world][helpers][equivalence][unit]") {
  SECTION("uniform(64) == create({64,64,64}, {0,0,0}, {1,1,1})") {
    auto w1 = world::uniform(64);
    auto w2 = world::create({64, 64, 64}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});

    REQUIRE(world::get_size(w1) == world::get_size(w2));
    REQUIRE(world::get_origin(w1) == world::get_origin(w2));
    REQUIRE(world::get_spacing(w1) == world::get_spacing(w2));
  }
}
