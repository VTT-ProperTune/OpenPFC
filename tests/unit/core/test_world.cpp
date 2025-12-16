// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <sstream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "openpfc/core/strong_types.hpp"
#include "openpfc/core/world.hpp"

using namespace Catch::Matchers;
using namespace pfc;
using namespace pfc::types;

TEST_CASE("World - construction and accessors", "[world][unit]") {
  SECTION("Construct World with valid dimensions, origin, and spacing") {
    Int3 dimensions = {100, 200, 300};
    Real3 origin = {1.0, 2.0, 3.0};
    Real3 spacing = {0.1, 0.2, 0.3};

    World world = world::create(GridSize(dimensions), PhysicalOrigin(origin),
                                GridSpacing(spacing));

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
    REQUIRE_THROWS_AS(world::create(GridSize({-1, 100, 100})),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(world::create(GridSize({0, 100, 100})), std::invalid_argument);
  }

  SECTION("Invalid spacing throws exception") {
    REQUIRE_THROWS_AS(world::create(GridSize({100, 100, 100}),
                                    PhysicalOrigin({0.0, 0.0, 0.0}),
                                    GridSpacing({-1.0, 1.0, 1.0})),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(world::create(GridSize({100, 100, 100}),
                                    PhysicalOrigin({0.0, 0.0, 0.0}),
                                    GridSpacing({0.0, 1.0, 1.0})),
                      std::invalid_argument);
  }
}

TEST_CASE("World - coordinate transformations", "[world][unit]") {
  Int3 dimensions = {100, 100, 100};
  Real3 origin = {0.0, 0.0, 0.0};
  Real3 spacing = {1.0, 1.0, 1.0};
  World world = world::create(GridSize(dimensions), PhysicalOrigin(origin),
                              GridSpacing(spacing));

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

  World world1 = world::create(GridSize(dimensions), PhysicalOrigin(origin),
                               GridSpacing(spacing));
  World world2 = world::create(GridSize(dimensions), PhysicalOrigin(origin),
                               GridSpacing(spacing));
  World world3 = world::create(GridSize({200, 100, 100}), PhysicalOrigin(origin),
                               GridSpacing(spacing));

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
    World world = world::create(GridSize(dimensions), PhysicalOrigin(origin),
                                GridSpacing(spacing));

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
    World world = world::create(GridSize({5, 5, 5}), PhysicalOrigin({0.0, 0.0, 0.0}),
                                GridSpacing({1.0, 1.0, 1.0}));
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
    World world = world::create(GridSize({8, 16, 32}), PhysicalOrigin(origin),
                                GridSpacing(spacing));

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
    auto w2 = world::create(GridSize({64, 64, 64}), PhysicalOrigin({0.0, 0.0, 0.0}),
                            GridSpacing({1.0, 1.0, 1.0}));

    REQUIRE(world::get_size(w1) == world::get_size(w2));
    REQUIRE(world::get_origin(w1) == world::get_origin(w2));
    REQUIRE(world::get_spacing(w1) == world::get_spacing(w2));
  }
}

TEST_CASE("World - physical_volume() calculates correctly",
          "[world][convenience][unit]") {
  SECTION("3D domain with unit spacing") {
    auto world =
        world::create(GridSize({10, 10, 10}), PhysicalOrigin({0.0, 0.0, 0.0}),
                      GridSpacing({1.0, 1.0, 1.0}));
    REQUIRE_THAT(world::physical_volume(world), WithinAbs(1000.0, 1e-10));
  }

  SECTION("3D domain with custom spacing") {
    auto world =
        world::create(GridSize({10, 10, 10}), PhysicalOrigin({0.0, 0.0, 0.0}),
                      GridSpacing({0.1, 0.1, 0.1}));
    REQUIRE_THAT(world::physical_volume(world), WithinAbs(1.0, 1e-10));
  }

  SECTION("2D domain (nz = 1)") {
    auto world =
        world::create(GridSize({100, 100, 1}), PhysicalOrigin({0.0, 0.0, 0.0}),
                      GridSpacing({0.01, 0.01, 1.0}));
    // Volume = 100 * 0.01 * 100 * 0.01 * 1 * 1.0 = 1.0
    REQUIRE_THAT(world::physical_volume(world), WithinAbs(1.0, 1e-10));
  }

  SECTION("1D domain (ny = nz = 1)") {
    auto world =
        world::create(GridSize({100, 1, 1}), PhysicalOrigin({0.0, 0.0, 0.0}),
                      GridSpacing({0.1, 1.0, 1.0}));
    // Volume = 100 * 0.1 * 1 * 1.0 * 1 * 1.0 = 10.0
    REQUIRE_THAT(world::physical_volume(world), WithinAbs(10.0, 1e-10));
  }

  SECTION("Non-cubic domain") {
    auto world =
        world::create(GridSize({128, 128, 32}), PhysicalOrigin({0.0, 0.0, 0.0}),
                      GridSpacing({0.01, 0.01, 0.05}));
    // Volume = 128 * 0.01 * 128 * 0.01 * 32 * 0.05 = 2.62144
    REQUIRE_THAT(world::physical_volume(world), WithinAbs(2.62144, 1e-10));
  }
}

TEST_CASE("World - dimensionality checks work correctly",
          "[world][convenience][unit]") {
  SECTION("1D domain (nx > 1, ny = 1, nz = 1)") {
    auto world1d = world::create(GridSize({100), PhysicalOrigin(1), GridSpacing(1}));

    REQUIRE(world::is_1d(world1d));
    REQUIRE_FALSE(world::is_2d(world1d));
    REQUIRE_FALSE(world::is_3d(world1d));
    REQUIRE(world::dimensionality(world1d) == 1);
  }

  SECTION("2D domain (nx > 1, ny > 1, nz = 1)") {
    auto world2d = world::create(GridSize({64), PhysicalOrigin(64), GridSpacing(1}));

    REQUIRE_FALSE(world::is_1d(world2d));
    REQUIRE(world::is_2d(world2d));
    REQUIRE_FALSE(world::is_3d(world2d));
    REQUIRE(world::dimensionality(world2d) == 2);
  }

  SECTION("3D domain (nx > 1, ny > 1, nz > 1)") {
    auto world3d = world::create(GridSize({32), PhysicalOrigin(32), GridSpacing(32}));

    REQUIRE_FALSE(world::is_1d(world3d));
    REQUIRE_FALSE(world::is_2d(world3d));
    REQUIRE(world::is_3d(world3d));
    REQUIRE(world::dimensionality(world3d) == 3);
  }

  SECTION("Degenerate case (nx = 1, ny = 1, nz = 1)") {
    auto world_degenerate = world::create(GridSize({1), PhysicalOrigin(1), GridSpacing(1}));

    REQUIRE_FALSE(world::is_1d(world_degenerate));
    REQUIRE_FALSE(world::is_2d(world_degenerate));
    REQUIRE_FALSE(world::is_3d(world_degenerate));
    REQUIRE(world::dimensionality(world_degenerate) == 0);
  }

  SECTION("Non-standard 2D (nx = 1, ny > 1, nz > 1)") {
    // This is NOT considered 2D by our definition
    auto world_yz = world::create(GridSize({1), PhysicalOrigin(64), GridSpacing(64}));

    REQUIRE_FALSE(world::is_1d(world_yz));
    REQUIRE_FALSE(world::is_2d(world_yz)); // Only x-y plane is 2D
    REQUIRE_FALSE(world::is_3d(world_yz));
    REQUIRE(world::dimensionality(world_yz) == 0); // Degenerate by our definition
  }
}

TEST_CASE("World - bounds accessors return correct values",
          "[world][convenience][unit]") {
  SECTION("Unit spacing at origin") {
    auto world =
        world::create(GridSize({10, 10, 10}), PhysicalOrigin({0.0, 0.0, 0.0}),
                      GridSpacing({1.0, 1.0, 1.0}));

    auto lower = world::get_lower_bounds(world);
    REQUIRE_THAT(lower[0], WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(lower[1], WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(lower[2], WithinAbs(0.0, 1e-10));

    auto upper = world::get_upper_bounds(world);
    REQUIRE_THAT(upper[0], WithinAbs(9.0, 1e-10)); // (10-1) * 1.0
    REQUIRE_THAT(upper[1], WithinAbs(9.0, 1e-10));
    REQUIRE_THAT(upper[2], WithinAbs(9.0, 1e-10));
  }

  SECTION("Custom spacing and origin") {
    auto world =
        world::create(GridSize({100, 100, 100}), PhysicalOrigin({-5.0, -5.0, 0.0}),
                      GridSpacing({0.1, 0.1, 0.1}));

    auto lower = world::get_lower_bounds(world);
    REQUIRE_THAT(lower[0], WithinAbs(-5.0, 1e-10));
    REQUIRE_THAT(lower[1], WithinAbs(-5.0, 1e-10));
    REQUIRE_THAT(lower[2], WithinAbs(0.0, 1e-10));

    auto upper = world::get_upper_bounds(world);
    REQUIRE_THAT(upper[0], WithinAbs(-5.0 + 99 * 0.1, 1e-10)); // -5.0 + 9.9 = 4.9
    REQUIRE_THAT(upper[1], WithinAbs(-5.0 + 99 * 0.1, 1e-10));
    REQUIRE_THAT(upper[2], WithinAbs(0.0 + 99 * 0.1, 1e-10)); // 9.9
  }

  SECTION("2D domain") {
    auto world =
        world::create(GridSize({128, 128, 1}), PhysicalOrigin({0.0, 0.0, 0.0}),
                      GridSpacing({0.01, 0.01, 1.0}));

    auto lower = world::get_lower_bounds(world);
    REQUIRE_THAT(lower[0], WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(lower[1], WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(lower[2], WithinAbs(0.0, 1e-10));

    auto upper = world::get_upper_bounds(world);
    REQUIRE_THAT(upper[0], WithinAbs(127 * 0.01, 1e-10)); // 1.27
    REQUIRE_THAT(upper[1], WithinAbs(127 * 0.01, 1e-10));
    REQUIRE_THAT(upper[2], WithinAbs(0.0, 1e-10)); // (1-1) * 1.0 = 0
  }

  SECTION("Bounds span equals expected physical size") {
    auto world =
        world::create(GridSize({50, 50, 50}), PhysicalOrigin({0.0, 0.0, 0.0}),
                      GridSpacing({0.2, 0.2, 0.2}));

    auto lower = world::get_lower_bounds(world);
    auto upper = world::get_upper_bounds(world);

    // Physical extent should be (size - 1) * spacing
    double extent_x = upper[0] - lower[0];
    double extent_y = upper[1] - lower[1];
    double extent_z = upper[2] - lower[2];

    REQUIRE_THAT(extent_x, WithinAbs(49 * 0.2, 1e-10)); // 9.8
    REQUIRE_THAT(extent_y, WithinAbs(49 * 0.2, 1e-10));
    REQUIRE_THAT(extent_z, WithinAbs(49 * 0.2, 1e-10));
  }
}

TEST_CASE("World - convenience functions work via ADL",
          "[world][convenience][adl][unit]") {
  using namespace world;

  SECTION("Functions accessible without world:: prefix") {
    auto w = create({64, 64, 64});

    // All these should work via ADL (no world:: prefix needed)
    auto vol = physical_volume(w);
    bool threed = is_3d(w);
    int dim = dimensionality(w);
    auto lower = get_lower_bounds(w);
    auto upper = get_upper_bounds(w);

    REQUIRE(vol > 0.0);
    REQUIRE(threed);
    REQUIRE(dim == 3);
    REQUIRE(lower[0] == 0.0);
    REQUIRE(upper[0] > 0.0);
  }
}

TEST_CASE("World - convenience functions integrate with existing API",
          "[world][convenience][integration][unit]") {
  SECTION("Physical volume matches manual calculation") {
    auto world =
        world::create(GridSize({100, 100, 100}), PhysicalOrigin({0.0, 0.0, 0.0}),
                      GridSpacing({0.1, 0.1, 0.1}));

    // Manual calculation
    auto spacing = world::get_spacing(world);
    auto size = world::get_size(world);
    double manual_vol =
        spacing[0] * spacing[1] * spacing[2] * size[0] * size[1] * size[2];

    // Using convenience function
    double conv_vol = world::physical_volume(world);

    REQUIRE_THAT(conv_vol, WithinAbs(manual_vol, 1e-10));
  }

  SECTION("Bounds match coordinate transformation results") {
    auto world =
        world::create(GridSize({64, 64, 64}), PhysicalOrigin({1.0, 2.0, 3.0}),
                      GridSpacing({0.5, 0.5, 0.5}));

    // Using convenience functions
    auto lower_conv = world::get_lower_bounds(world);
    auto upper_conv = world::get_upper_bounds(world);

    // Using existing coordinate transformation
    auto lower_manual = world::to_coords(world, {0, 0, 0});
    auto size = world::get_size(world);
    auto upper_manual =
        world::to_coords(world, {size[0] - 1, size[1] - 1, size[2] - 1});

    REQUIRE(lower_conv == lower_manual);
    REQUIRE(upper_conv == upper_manual);
  }

  SECTION("Dimensionality checks consistent with size queries") {
    auto world2d = world::create(GridSize({128), PhysicalOrigin(128), GridSpacing(1}));

    REQUIRE(world::is_2d(world2d));
    REQUIRE(world::get_size(world2d, 0) > 1);
    REQUIRE(world::get_size(world2d, 1) > 1);
    REQUIRE(world::get_size(world2d, 2) == 1);
    REQUIRE(world::dimensionality(world2d) == 2);
  }
}
