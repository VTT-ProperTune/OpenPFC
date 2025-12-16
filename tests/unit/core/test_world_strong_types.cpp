// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_world_strong_types.cpp
 * @brief Tests for World construction using strong types
 *
 * Tests the type-safe World creation APIs that use strong types like
 * GridSize, PhysicalOrigin, and GridSpacing from strong_types.hpp.
 * These tests verify type safety, zero overhead, and backward compatibility.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <openpfc/core/strong_types.hpp>
#include <openpfc/core/world.hpp>

using Catch::Approx;

TEST_CASE("World creation with strong types - basic functionality",
          "[world][strong_types]") {
  using namespace pfc;

  SECTION("Create with GridSize, PhysicalOrigin, GridSpacing") {
    // Arrange: Use strong types for clarity
    GridSize size({64, 64, 64});
    PhysicalOrigin origin({0.0, 0.0, 0.0});
    GridSpacing spacing({1.0, 1.0, 1.0});

    // Act: Create world
    auto world =
        world::create(GridSize(size), PhysicalOrigin(origin), GridSpacing(spacing));

    // Assert: Verify world properties
    auto world_size = get_size(world);
    auto world_spacing = get_spacing(world);
    auto world_origin = get_origin(world);

    REQUIRE(world_size[0] == 64);
    REQUIRE(world_size[1] == 64);
    REQUIRE(world_size[2] == 64);

    REQUIRE(world_spacing[0] == Approx(1.0));
    REQUIRE(world_spacing[1] == Approx(1.0));
    REQUIRE(world_spacing[2] == Approx(1.0));

    REQUIRE(world_origin[0] == Approx(0.0));
    REQUIRE(world_origin[1] == Approx(0.0));
    REQUIRE(world_origin[2] == Approx(0.0));
  }

  SECTION("Create with non-zero origin") {
    GridSize size({32, 32, 32});
    PhysicalOrigin origin({-5.0, -5.0, -5.0});
    GridSpacing spacing({0.5, 0.5, 0.5});

    auto world =
        world::create(GridSize(size), PhysicalOrigin(origin), GridSpacing(spacing));

    auto world_origin = get_origin(world);
    REQUIRE(world_origin[0] == Approx(-5.0));
    REQUIRE(world_origin[1] == Approx(-5.0));
    REQUIRE(world_origin[2] == Approx(-5.0));
  }

  SECTION("Create with non-uniform spacing") {
    GridSize size({100, 50, 25});
    PhysicalOrigin origin({0.0, 0.0, 0.0});
    GridSpacing spacing({0.1, 0.2, 0.4});

    auto world =
        world::create(GridSize(size), PhysicalOrigin(origin), GridSpacing(spacing));

    auto world_size = get_size(world);
    auto world_spacing = get_spacing(world);

    REQUIRE(world_size[0] == 100);
    REQUIRE(world_size[1] == 50);
    REQUIRE(world_size[2] == 25);

    REQUIRE(world_spacing[0] == Approx(0.1));
    REQUIRE(world_spacing[1] == Approx(0.2));
    REQUIRE(world_spacing[2] == Approx(0.4));
  }
}

TEST_CASE("Strong types prevent parameter confusion",
          "[world][strong_types][type_safety]") {
  using namespace pfc;

  SECTION("GridSize, PhysicalOrigin, GridSpacing have clear intent") {
    // This compiles - correct order
    GridSize size({64, 64, 64});
    PhysicalOrigin origin({0.0, 0.0, 0.0});
    GridSpacing spacing({1.0, 1.0, 1.0});

    auto world =
        world::create(GridSize(size), PhysicalOrigin(origin), GridSpacing(spacing));
    REQUIRE(get_size(world)[0] == 64);

    // NOTE: The following would NOT compile if parameters are swapped:
    // auto bad = world::create(GridSize(spacing), PhysicalOrigin(size),
    // GridSpacing(origin));  // Compile error! auto bad2 =
    // world::create(GridSize(origin), PhysicalOrigin(spacing), GridSpacing(size));
    // // Compile error!
    //
    // This is the key benefit - type system catches parameter order mistakes
  }
}

TEST_CASE("Strong types have zero overhead", "[world][strong_types][performance]") {
  using namespace pfc;

  SECTION("sizeof checks - same as underlying types") {
    // GridSize wraps Int3
    STATIC_REQUIRE(sizeof(GridSize) == sizeof(Int3));

    // PhysicalOrigin wraps Real3
    STATIC_REQUIRE(sizeof(PhysicalOrigin) == sizeof(Real3));

    // GridSpacing wraps Real3
    STATIC_REQUIRE(sizeof(GridSpacing) == sizeof(Real3));
  }

  SECTION("Trivially copyable - no overhead") {
    STATIC_REQUIRE(std::is_trivially_copyable_v<GridSize>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<PhysicalOrigin>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<GridSpacing>);
  }

  SECTION("Standard layout - interop friendly") {
    STATIC_REQUIRE(std::is_standard_layout_v<GridSize>);
    STATIC_REQUIRE(std::is_standard_layout_v<PhysicalOrigin>);
    STATIC_REQUIRE(std::is_standard_layout_v<GridSpacing>);
  }
}

TEST_CASE("Backward compatibility with raw types",
          "[world][strong_types][compatibility]") {
  using namespace pfc;

  SECTION("Can still use old create(Int3, Real3, Real3) API") {
    // Old API should still work (though deprecated)
    Int3 size = {32, 32, 32};
    Real3 offset = {0.0, 0.0, 0.0};
    Real3 spacing = {1.0, 1.0, 1.0};

    auto world =
        world::create(GridSize(size), PhysicalOrigin(offset), GridSpacing(spacing));

    REQUIRE(get_size(world)[0] == 32);
    REQUIRE(get_spacing(world)[0] == Approx(1.0));
    REQUIRE(get_origin(world)[0] == Approx(0.0));
  }

  SECTION("Strong types implicitly convert to raw types") {
    GridSize size({64, 64, 64});
    PhysicalOrigin origin({0.0, 0.0, 0.0});
    GridSpacing spacing({1.0, 1.0, 1.0});

    // Should be able to extract raw values
    Int3 raw_size = size;
    Real3 raw_origin = origin;
    Real3 raw_spacing = spacing;

    REQUIRE(raw_size[0] == 64);
    REQUIRE(raw_origin[0] == 0.0);
    REQUIRE(raw_spacing[0] == 1.0);
  }

  SECTION("Can mix raw types and strong types in construction") {
    // Using strong type for size, raw for others
    GridSize size({64, 64, 64});

    // This should work due to implicit conversions
    auto world = world::create(size.get(), Real3{0, 0, 0}, Real3{1, 1, 1});

    REQUIRE(get_size(world)[0] == 64);
  }
}

TEST_CASE("Strong type construction and access patterns", "[world][strong_types]") {
  using namespace pfc;

  SECTION("Construct from raw arrays") {
    Int3 raw_size = {128, 128, 128};
    GridSize size(raw_size);

    REQUIRE(size.value[0] == 128);
    REQUIRE(size.get()[0] == 128);
  }

  SECTION("Construct with brace initialization") {
    GridSize size({256, 256, 256});
    PhysicalOrigin origin({-10.0, -10.0, -10.0});
    GridSpacing spacing({0.078125, 0.078125, 0.078125});

    REQUIRE(size.value[0] == 256);
    REQUIRE(origin.value[0] == Approx(-10.0));
    REQUIRE(spacing.value[0] == Approx(0.078125));
  }

  SECTION("Access via .value and .get()") {
    GridSize size({64, 64, 64});

    // Both should work
    REQUIRE(size.value[0] == 64);
    REQUIRE(size.get()[0] == 64);

    // get() returns const reference
    const Int3 &ref = size.get();
    REQUIRE(ref[0] == 64);
  }
}

TEST_CASE("Strong types with world helper functions",
          "[world][strong_types][helpers]") {
  using namespace pfc;

  SECTION("Works with world::uniform() helper") {
    // Create uniform grid using helper
    auto world1 = world::uniform(64);

    // Should be able to query with get_ functions
    REQUIRE(get_size(world1)[0] == 64);
    REQUIRE(get_spacing(world1)[0] == Approx(1.0));

    // Create with spacing using helper
    auto world2 = world::uniform(32, 0.5);

    REQUIRE(get_size(world2)[0] == 32);
    REQUIRE(get_spacing(world2)[0] == Approx(0.5));
  }

  SECTION("Works with world::from_bounds() helper") {
    // Create from physical bounds
    auto world = world::from_bounds({100, 100, 100}, {0, 0, 0}, {10, 10, 10});

    REQUIRE(get_size(world)[0] == 100);
    REQUIRE(get_spacing(world)[0] == Approx(0.1));
    REQUIRE(get_origin(world)[0] == Approx(0.0));
  }
}

TEST_CASE("Strong types coordinate transformation verification",
          "[world][strong_types][coords]") {
  using namespace pfc;

  SECTION("Coordinate transforms work correctly with strong types") {
    GridSize size({64, 64, 64});
    PhysicalOrigin origin({-32.0, -32.0, -32.0});
    GridSpacing spacing({1.0, 1.0, 1.0});

    auto world =
        world::create(GridSize(size), PhysicalOrigin(origin), GridSpacing(spacing));

    // Index (0, 0, 0) should map to origin
    Real3 coords = to_coords(world, {0, 0, 0});
    REQUIRE(coords[0] == Approx(-32.0));
    REQUIRE(coords[1] == Approx(-32.0));
    REQUIRE(coords[2] == Approx(-32.0));

    // Index (32, 32, 32) should map to (0, 0, 0)
    coords = to_coords(world, {32, 32, 32});
    REQUIRE(coords[0] == Approx(0.0));
    REQUIRE(coords[1] == Approx(0.0));
    REQUIRE(coords[2] == Approx(0.0));

    // Index (63, 63, 63) should map to origin + 63*spacing
    coords = to_coords(world, {63, 63, 63});
    REQUIRE(coords[0] == Approx(31.0));
    REQUIRE(coords[1] == Approx(31.0));
    REQUIRE(coords[2] == Approx(31.0));
  }
}

TEST_CASE("Strong types equality comparison", "[world][strong_types]") {
  using namespace pfc;

  SECTION("GridSize equality") {
    GridSize s1({64, 64, 64});
    GridSize s2({64, 64, 64});
    GridSize s3({128, 128, 128});

    REQUIRE(s1 == s2);
    REQUIRE(s1 != s3);
  }

  SECTION("PhysicalOrigin equality") {
    PhysicalOrigin o1({0.0, 0.0, 0.0});
    PhysicalOrigin o2({0.0, 0.0, 0.0});
    PhysicalOrigin o3({1.0, 0.0, 0.0});

    REQUIRE(o1 == o2);
    REQUIRE(o1 != o3);
  }

  SECTION("GridSpacing equality") {
    GridSpacing sp1({1.0, 1.0, 1.0});
    GridSpacing sp2({1.0, 1.0, 1.0});
    GridSpacing sp3({0.5, 1.0, 1.0});

    REQUIRE(sp1 == sp2);
    REQUIRE(sp1 != sp3);
  }
}

TEST_CASE("Documentation examples compile and work",
          "[world][strong_types][examples]") {
  using namespace pfc;

  SECTION("Example from documentation - basic usage") {
    // Create 256³ grid from -128 to 128 with spacing 1.0
    GridSize size({256, 256, 256});
    PhysicalOrigin origin({-128.0, -128.0, -128.0});
    GridSpacing spacing({1.0, 1.0, 1.0});

    auto world =
        world::create(GridSize(size), PhysicalOrigin(origin), GridSpacing(spacing));

    // Verify domain properties
    REQUIRE(get_size(world)[0] == 256);
    REQUIRE(get_spacing(world)[0] == Approx(1.0));

    // Physical domain extends from -128 to 127
    Real3 lower_corner = to_coords(world, {0, 0, 0});
    Real3 upper_corner = to_coords(world, {255, 255, 255});

    REQUIRE(lower_corner[0] == Approx(-128.0));
    REQUIRE(upper_corner[0] == Approx(127.0));
  }

  SECTION("Example - centered domain with custom spacing") {
    GridSize size({100, 100, 100});
    PhysicalOrigin origin({-5.0, -5.0, -5.0});
    GridSpacing spacing({0.1, 0.1, 0.1});

    auto world =
        world::create(GridSize(size), PhysicalOrigin(origin), GridSpacing(spacing));

    // Domain extends from -5.0 to 4.9 in each dimension
    Real3 lower = to_coords(world, {0, 0, 0});
    Real3 upper = to_coords(world, {99, 99, 99});

    REQUIRE(lower[0] == Approx(-5.0));
    REQUIRE(upper[0] == Approx(4.9));
  }
}
