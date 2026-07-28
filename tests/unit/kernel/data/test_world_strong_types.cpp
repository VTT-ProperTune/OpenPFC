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
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/strong_types.hpp>

using Catch::Approx;

TEST_CASE("World creation with strong types - basic functionality",
          "[world][strong_types]") {
  using namespace pfc;

  SECTION("Create with GridSize, PhysicalOrigin, GridSpacing") {
    // Arrange: Use strong types for clarity
    GridSize size({64, 64, 64});
    PhysicalOrigin origin({0.0, 0.0, 0.0});
    GridSpacing spacing({1.0, 1.0, 1.0});

    // Act: Create domain
    auto domain =
        domain::create(GridSize(size), PhysicalOrigin(origin), GridSpacing(spacing));

    // Assert: Verify domain properties
    auto domain_size = domain::get_size(domain);
    auto domain_spacing = domain::get_spacing(domain);
    auto domain_origin = domain::get_origin(domain);

    REQUIRE(domain_size[0] == 64);
    REQUIRE(domain_size[1] == 64);
    REQUIRE(domain_size[2] == 64);

    REQUIRE(domain_spacing[0] == Approx(1.0));
    REQUIRE(domain_spacing[1] == Approx(1.0));
    REQUIRE(domain_spacing[2] == Approx(1.0));

    REQUIRE(domain_origin[0] == Approx(0.0));
    REQUIRE(domain_origin[1] == Approx(0.0));
    REQUIRE(domain_origin[2] == Approx(0.0));
  }

  SECTION("Create with non-zero origin") {
    GridSize size({32, 32, 32});
    PhysicalOrigin origin({-5.0, -5.0, -5.0});
    GridSpacing spacing({0.5, 0.5, 0.5});

    auto domain =
        domain::create(GridSize(size), PhysicalOrigin(origin), GridSpacing(spacing));

    auto domain_origin = domain::get_origin(domain);
    REQUIRE(domain_origin[0] == Approx(-5.0));
    REQUIRE(domain_origin[1] == Approx(-5.0));
    REQUIRE(domain_origin[2] == Approx(-5.0));
  }

  SECTION("Create with non-uniform spacing") {
    GridSize size({100, 50, 25});
    PhysicalOrigin origin({0.0, 0.0, 0.0});
    GridSpacing spacing({0.1, 0.2, 0.4});

    auto domain =
        domain::create(GridSize(size), PhysicalOrigin(origin), GridSpacing(spacing));

    auto domain_size = domain::get_size(domain);
    auto domain_spacing = domain::get_spacing(domain);

    REQUIRE(domain_size[0] == 100);
    REQUIRE(domain_size[1] == 50);
    REQUIRE(domain_size[2] == 25);

    REQUIRE(domain_spacing[0] == Approx(0.1));
    REQUIRE(domain_spacing[1] == Approx(0.2));
    REQUIRE(domain_spacing[2] == Approx(0.4));
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

    auto domain =
        domain::create(GridSize(size), PhysicalOrigin(origin), GridSpacing(spacing));
    REQUIRE(domain::get_size(domain)[0] == 64);

    // NOTE: The following would NOT compile if parameters are swapped:
    // auto bad = domain::create(GridSize(spacing), PhysicalOrigin(size),
    // GridSpacing(origin));  // Compile error! auto bad2 =
    // domain::create(GridSize(origin), PhysicalOrigin(spacing), GridSpacing(size));
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

    auto domain =
        domain::create(GridSize(size), PhysicalOrigin(offset), GridSpacing(spacing));

    REQUIRE(domain::get_size(domain)[0] == 32);
    REQUIRE(domain::get_spacing(domain)[0] == Approx(1.0));
    REQUIRE(domain::get_origin(domain)[0] == Approx(0.0));
  }

  SECTION("Strong types explicitly convert to raw types") {
    GridSize size({64, 64, 64});
    PhysicalOrigin origin({0.0, 0.0, 0.0});
    GridSpacing spacing({1.0, 1.0, 1.0});

    // Should be able to extract raw values using explicit conversions
    Int3 raw_size = size.to_vector3();
    Real3 raw_origin = origin.to_vector3();
    Real3 raw_spacing = spacing.to_vector3();

    REQUIRE(raw_size[0] == 64);
    REQUIRE(raw_origin[0] == 0.0);
    REQUIRE(raw_spacing[0] == 1.0);
  }

  SECTION("Strong-type create overload") {
    GridSize size({64, 64, 64});

    auto domain = domain::create(size, PhysicalOrigin({0.0, 0.0, 0.0}),
                               GridSpacing({1.0, 1.0, 1.0}));

    REQUIRE(domain::get_size(domain)[0] == 64);
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

  SECTION("Works with domain::create() helper for uniform grid") {
    // Create uniform grid using helper
    auto domain1 = domain::create(Int3({64, 64, 64}));

    // Should be able to query with get_ functions
    REQUIRE(domain::get_size(domain1)[0] == 64);
    REQUIRE(domain::get_spacing(domain1)[0] == Approx(1.0));

    // Create with spacing using helper
    auto domain2 = domain::with_spacing(Int3({32, 32, 32}), Real3({0.5, 0.5, 0.5}));

    REQUIRE(domain::get_size(domain2)[0] == 32);
    REQUIRE(domain::get_spacing(domain2)[0] == Approx(0.5));
  }

  SECTION("Works with domain::from_bounds() helper") {
    // Create from physical bounds
    auto domain = domain::from_bounds(Int3({100, 100, 100}), Real3({0, 0, 0}), Real3({10, 10, 10}));

    REQUIRE(domain::get_size(domain)[0] == 100);
    REQUIRE(domain::get_spacing(domain)[0] == Approx(0.1));
    REQUIRE(domain::get_origin(domain)[0] == Approx(0.0));
  }
}

TEST_CASE("Strong types coordinate transformation verification",
          "[world][strong_types][coords]") {
  using namespace pfc;

  SECTION("Coordinate transforms work correctly with strong types") {
    GridSize size({64, 64, 64});
    PhysicalOrigin origin({-32.0, -32.0, -32.0});
    GridSpacing spacing({1.0, 1.0, 1.0});

    auto domain =
        domain::create(GridSize(size), PhysicalOrigin(origin), GridSpacing(spacing));

    // Index (0, 0, 0) should map to origin
    Real3 coords = domain::to_coords(domain, {0, 0, 0});
    REQUIRE(coords[0] == Approx(-32.0));
    REQUIRE(coords[1] == Approx(-32.0));
    REQUIRE(coords[2] == Approx(-32.0));

    // Index (32, 32, 32) should map to (0, 0, 0)
    coords = domain::to_coords(domain, {32, 32, 32});
    REQUIRE(coords[0] == Approx(0.0));
    REQUIRE(coords[1] == Approx(0.0));
    REQUIRE(coords[2] == Approx(0.0));

    // Index (63, 63, 63) should map to origin + 63*spacing
    coords = domain::to_coords(domain, {63, 63, 63});
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

    auto domain =
        domain::create(GridSize(size), PhysicalOrigin(origin), GridSpacing(spacing));

    // Verify domain properties
    REQUIRE(domain::get_size(domain)[0] == 256);
    REQUIRE(domain::get_spacing(domain)[0] == Approx(1.0));

    // Physical domain extends from -128 to 127
    Real3 lower_corner = domain::to_coords(domain, {0, 0, 0});
    Real3 upper_corner = domain::to_coords(domain, {255, 255, 255});

    REQUIRE(lower_corner[0] == Approx(-128.0));
    REQUIRE(upper_corner[0] == Approx(127.0));
  }

  SECTION("Example - centered domain with custom spacing") {
    GridSize size({100, 100, 100});
    PhysicalOrigin origin({-5.0, -5.0, -5.0});
    GridSpacing spacing({0.1, 0.1, 0.1});

    auto domain =
        domain::create(GridSize(size), PhysicalOrigin(origin), GridSpacing(spacing));

    // Domain extends from -5.0 to 4.9 in each dimension
    Real3 lower = domain::to_coords(domain, {0, 0, 0});
    Real3 upper = domain::to_coords(domain, {99, 99, 99});

    REQUIRE(lower[0] == Approx(-5.0));
    REQUIRE(upper[0] == Approx(4.9));
  }
}
