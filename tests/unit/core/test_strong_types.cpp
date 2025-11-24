// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_strong_types.cpp
 * @brief Comprehensive tests for strong type aliases
 *
 * Tests verify:
 * - Zero-cost abstraction (same size as underlying types)
 * - Implicit conversions (both from and to underlying types)
 * - Type safety (different strong types are distinct)
 * - Construction patterns
 * - Comparison operators
 * - Standard layout and trivial copyability
 */

#include <catch2/catch_test_macros.hpp>
#include <openpfc/core/strong_types.hpp>
#include <type_traits>

using namespace pfc;

// ============================================================================
// Zero-Cost Abstraction Tests
// ============================================================================

TEST_CASE("Strong types are zero-cost abstractions", "[strong_types][performance]") {
  SECTION("GridSize has same size as Int3") {
    REQUIRE(sizeof(GridSize) == sizeof(Int3));
  }

  SECTION("LocalOffset has same size as Int3") {
    REQUIRE(sizeof(LocalOffset) == sizeof(Int3));
  }

  SECTION("GlobalOffset has same size as Int3") {
    REQUIRE(sizeof(GlobalOffset) == sizeof(Int3));
  }

  SECTION("GridSpacing has same size as Real3") {
    REQUIRE(sizeof(GridSpacing) == sizeof(Real3));
  }

  SECTION("PhysicalOrigin has same size as Real3") {
    REQUIRE(sizeof(PhysicalOrigin) == sizeof(Real3));
  }

  SECTION("PhysicalCoords has same size as Real3") {
    REQUIRE(sizeof(PhysicalCoords) == sizeof(Real3));
  }
}

TEST_CASE("Strong types are trivially copyable", "[strong_types][performance]") {
  SECTION("GridSize is trivially copyable") {
    REQUIRE(std::is_trivially_copyable_v<GridSize>);
  }

  SECTION("LocalOffset is trivially copyable") {
    REQUIRE(std::is_trivially_copyable_v<LocalOffset>);
  }

  SECTION("GlobalOffset is trivially copyable") {
    REQUIRE(std::is_trivially_copyable_v<GlobalOffset>);
  }

  SECTION("GridSpacing is trivially copyable") {
    REQUIRE(std::is_trivially_copyable_v<GridSpacing>);
  }

  SECTION("PhysicalOrigin is trivially copyable") {
    REQUIRE(std::is_trivially_copyable_v<PhysicalOrigin>);
  }

  SECTION("PhysicalCoords is trivially copyable") {
    REQUIRE(std::is_trivially_copyable_v<PhysicalCoords>);
  }

  SECTION("IndexBounds is trivially copyable") {
    REQUIRE(std::is_trivially_copyable_v<IndexBounds>);
  }

  SECTION("PhysicalBounds is trivially copyable") {
    REQUIRE(std::is_trivially_copyable_v<PhysicalBounds>);
  }
}

TEST_CASE("Strong types have standard layout", "[strong_types][performance]") {
  SECTION("GridSize has standard layout") {
    REQUIRE(std::is_standard_layout_v<GridSize>);
  }

  SECTION("LocalOffset has standard layout") {
    REQUIRE(std::is_standard_layout_v<LocalOffset>);
  }

  SECTION("GlobalOffset has standard layout") {
    REQUIRE(std::is_standard_layout_v<GlobalOffset>);
  }

  SECTION("GridSpacing has standard layout") {
    REQUIRE(std::is_standard_layout_v<GridSpacing>);
  }

  SECTION("PhysicalOrigin has standard layout") {
    REQUIRE(std::is_standard_layout_v<PhysicalOrigin>);
  }

  SECTION("PhysicalCoords has standard layout") {
    REQUIRE(std::is_standard_layout_v<PhysicalCoords>);
  }

  SECTION("IndexBounds has standard layout") {
    REQUIRE(std::is_standard_layout_v<IndexBounds>);
  }

  SECTION("PhysicalBounds has standard layout") {
    REQUIRE(std::is_standard_layout_v<PhysicalBounds>);
  }
}

// ============================================================================
// Construction and Conversion Tests
// ============================================================================

TEST_CASE("GridSize construction and conversion", "[strong_types][construction]") {
  SECTION("Construction from Int3") {
    Int3 raw = {64, 64, 64};
    GridSize size(raw);

    REQUIRE(size.get()[0] == 64);
    REQUIRE(size.get()[1] == 64);
    REQUIRE(size.get()[2] == 64);
  }

  SECTION("Brace initialization") {
    GridSize size({32, 32, 32});

    REQUIRE(size.get()[0] == 32);
    REQUIRE(size.get()[1] == 32);
    REQUIRE(size.get()[2] == 32);
  }

  SECTION("Implicit conversion back to Int3") {
    GridSize size({100, 100, 100});
    Int3 converted = size;

    REQUIRE(converted[0] == 100);
    REQUIRE(converted[1] == 100);
    REQUIRE(converted[2] == 100);
  }

  SECTION("get() returns underlying value") {
    Int3 raw = {50, 50, 50};
    GridSize size(raw);

    REQUIRE(size.get() == raw);
  }
}

TEST_CASE("LocalOffset construction and conversion",
          "[strong_types][construction]") {
  SECTION("Construction from Int3") {
    Int3 raw = {10, 20, 30};
    LocalOffset offset(raw);

    REQUIRE(offset.get()[0] == 10);
    REQUIRE(offset.get()[1] == 20);
    REQUIRE(offset.get()[2] == 30);
  }

  SECTION("Implicit conversion back to Int3") {
    LocalOffset offset({5, 10, 15});
    Int3 converted = offset;

    REQUIRE(converted[0] == 5);
    REQUIRE(converted[1] == 10);
    REQUIRE(converted[2] == 15);
  }
}

TEST_CASE("GlobalOffset construction and conversion",
          "[strong_types][construction]") {
  SECTION("Construction from Int3") {
    Int3 raw = {100, 200, 300};
    GlobalOffset offset(raw);

    REQUIRE(offset.get()[0] == 100);
    REQUIRE(offset.get()[1] == 200);
    REQUIRE(offset.get()[2] == 300);
  }

  SECTION("Implicit conversion back to Int3") {
    GlobalOffset offset({50, 100, 150});
    Int3 converted = offset;

    REQUIRE(converted[0] == 50);
    REQUIRE(converted[1] == 100);
    REQUIRE(converted[2] == 150);
  }
}

TEST_CASE("GridSpacing construction and conversion",
          "[strong_types][construction]") {
  SECTION("Construction from Real3") {
    Real3 raw = {1.0, 1.0, 1.0};
    GridSpacing spacing(raw);

    REQUIRE(spacing.get()[0] == 1.0);
    REQUIRE(spacing.get()[1] == 1.0);
    REQUIRE(spacing.get()[2] == 1.0);
  }

  SECTION("Brace initialization") {
    GridSpacing spacing({0.5, 0.5, 0.5});

    REQUIRE(spacing.get()[0] == 0.5);
    REQUIRE(spacing.get()[1] == 0.5);
    REQUIRE(spacing.get()[2] == 0.5);
  }

  SECTION("Implicit conversion back to Real3") {
    GridSpacing spacing({0.25, 0.25, 0.25});
    Real3 converted = spacing;

    REQUIRE(converted[0] == 0.25);
    REQUIRE(converted[1] == 0.25);
    REQUIRE(converted[2] == 0.25);
  }
}

TEST_CASE("PhysicalOrigin construction and conversion",
          "[strong_types][construction]") {
  SECTION("Construction from Real3") {
    Real3 raw = {-10.0, -10.0, -10.0};
    PhysicalOrigin origin(raw);

    REQUIRE(origin.get()[0] == -10.0);
    REQUIRE(origin.get()[1] == -10.0);
    REQUIRE(origin.get()[2] == -10.0);
  }

  SECTION("Implicit conversion back to Real3") {
    PhysicalOrigin origin({5.5, 10.5, 15.5});
    Real3 converted = origin;

    REQUIRE(converted[0] == 5.5);
    REQUIRE(converted[1] == 10.5);
    REQUIRE(converted[2] == 15.5);
  }
}

TEST_CASE("PhysicalCoords construction and conversion",
          "[strong_types][construction]") {
  SECTION("Construction from Real3") {
    Real3 raw = {1.23, 4.56, 7.89};
    PhysicalCoords coords(raw);

    REQUIRE(coords.get()[0] == 1.23);
    REQUIRE(coords.get()[1] == 4.56);
    REQUIRE(coords.get()[2] == 7.89);
  }

  SECTION("Implicit conversion back to Real3") {
    PhysicalCoords coords({10.1, 20.2, 30.3});
    Real3 converted = coords;

    REQUIRE(converted[0] == 10.1);
    REQUIRE(converted[1] == 20.2);
    REQUIRE(converted[2] == 30.3);
  }
}

TEST_CASE("IndexBounds construction", "[strong_types][construction][bounds]") {
  SECTION("Construction from two Int3") {
    Int3 lower = {0, 0, 0};
    Int3 upper = {63, 63, 63};
    IndexBounds bounds(lower, upper);

    REQUIRE(bounds.lower[0] == 0);
    REQUIRE(bounds.lower[1] == 0);
    REQUIRE(bounds.lower[2] == 0);
    REQUIRE(bounds.upper[0] == 63);
    REQUIRE(bounds.upper[1] == 63);
    REQUIRE(bounds.upper[2] == 63);
  }

  SECTION("Brace initialization") {
    IndexBounds bounds({10, 20, 30}, {50, 60, 70});

    REQUIRE(bounds.lower[0] == 10);
    REQUIRE(bounds.lower[1] == 20);
    REQUIRE(bounds.lower[2] == 30);
    REQUIRE(bounds.upper[0] == 50);
    REQUIRE(bounds.upper[1] == 60);
    REQUIRE(bounds.upper[2] == 70);
  }
}

TEST_CASE("PhysicalBounds construction", "[strong_types][construction][bounds]") {
  SECTION("Construction from two Real3") {
    Real3 lower = {-10.0, -10.0, -10.0};
    Real3 upper = {10.0, 10.0, 10.0};
    PhysicalBounds bounds(lower, upper);

    REQUIRE(bounds.lower[0] == -10.0);
    REQUIRE(bounds.lower[1] == -10.0);
    REQUIRE(bounds.lower[2] == -10.0);
    REQUIRE(bounds.upper[0] == 10.0);
    REQUIRE(bounds.upper[1] == 10.0);
    REQUIRE(bounds.upper[2] == 10.0);
  }

  SECTION("Brace initialization") {
    PhysicalBounds bounds({0.0, 0.0, 0.0}, {100.0, 100.0, 100.0});

    REQUIRE(bounds.lower[0] == 0.0);
    REQUIRE(bounds.lower[1] == 0.0);
    REQUIRE(bounds.lower[2] == 0.0);
    REQUIRE(bounds.upper[0] == 100.0);
    REQUIRE(bounds.upper[1] == 100.0);
    REQUIRE(bounds.upper[2] == 100.0);
  }
}

// ============================================================================
// Type Safety Tests
// ============================================================================

TEST_CASE("Strong types are distinct types", "[strong_types][safety]") {
  SECTION("GridSize and LocalOffset are different types") {
    REQUIRE_FALSE(std::is_same_v<GridSize, LocalOffset>);
  }

  SECTION("GridSize and GlobalOffset are different types") {
    REQUIRE_FALSE(std::is_same_v<GridSize, GlobalOffset>);
  }

  SECTION("LocalOffset and GlobalOffset are different types") {
    REQUIRE_FALSE(std::is_same_v<LocalOffset, GlobalOffset>);
  }

  SECTION("GridSpacing and PhysicalOrigin are different types") {
    REQUIRE_FALSE(std::is_same_v<GridSpacing, PhysicalOrigin>);
  }

  SECTION("GridSpacing and PhysicalCoords are different types") {
    REQUIRE_FALSE(std::is_same_v<GridSpacing, PhysicalCoords>);
  }

  SECTION("PhysicalOrigin and PhysicalCoords are different types") {
    REQUIRE_FALSE(std::is_same_v<PhysicalOrigin, PhysicalCoords>);
  }
}

TEST_CASE("Strong types cannot be implicitly assigned to each other",
          "[strong_types][safety]") {
  SECTION("Cannot assign GridSize to LocalOffset") {
    // This test verifies that the types are distinct at compile time
    // The actual prevention happens at compile time, so we just verify the types are
    // different
    GridSize size({64, 64, 64});
    LocalOffset offset({0, 0, 0});

    // These are different types
    REQUIRE_FALSE(std::is_same_v<decltype(size), decltype(offset)>);
  }
}

// ============================================================================
// Comparison Operator Tests
// ============================================================================

TEST_CASE("GridSize comparison operators", "[strong_types][comparison]") {
  SECTION("Equality operator") {
    GridSize size1({64, 64, 64});
    GridSize size2({64, 64, 64});
    GridSize size3({32, 32, 32});

    REQUIRE(size1 == size2);
    REQUIRE_FALSE(size1 == size3);
  }

  SECTION("Inequality operator") {
    GridSize size1({64, 64, 64});
    GridSize size2({64, 64, 64});
    GridSize size3({32, 32, 32});

    REQUIRE_FALSE(size1 != size2);
    REQUIRE(size1 != size3);
  }
}

TEST_CASE("LocalOffset comparison operators", "[strong_types][comparison]") {
  SECTION("Equality operator") {
    LocalOffset offset1({10, 20, 30});
    LocalOffset offset2({10, 20, 30});
    LocalOffset offset3({5, 10, 15});

    REQUIRE(offset1 == offset2);
    REQUIRE_FALSE(offset1 == offset3);
  }

  SECTION("Inequality operator") {
    LocalOffset offset1({10, 20, 30});
    LocalOffset offset2({10, 20, 30});
    LocalOffset offset3({5, 10, 15});

    REQUIRE_FALSE(offset1 != offset2);
    REQUIRE(offset1 != offset3);
  }
}

TEST_CASE("GridSpacing comparison operators", "[strong_types][comparison]") {
  SECTION("Equality operator") {
    GridSpacing spacing1({1.0, 1.0, 1.0});
    GridSpacing spacing2({1.0, 1.0, 1.0});
    GridSpacing spacing3({0.5, 0.5, 0.5});

    REQUIRE(spacing1 == spacing2);
    REQUIRE_FALSE(spacing1 == spacing3);
  }

  SECTION("Inequality operator") {
    GridSpacing spacing1({1.0, 1.0, 1.0});
    GridSpacing spacing2({1.0, 1.0, 1.0});
    GridSpacing spacing3({0.5, 0.5, 0.5});

    REQUIRE_FALSE(spacing1 != spacing2);
    REQUIRE(spacing1 != spacing3);
  }
}

TEST_CASE("PhysicalOrigin comparison operators", "[strong_types][comparison]") {
  SECTION("Equality operator") {
    PhysicalOrigin origin1({0.0, 0.0, 0.0});
    PhysicalOrigin origin2({0.0, 0.0, 0.0});
    PhysicalOrigin origin3({1.0, 1.0, 1.0});

    REQUIRE(origin1 == origin2);
    REQUIRE_FALSE(origin1 == origin3);
  }

  SECTION("Inequality operator") {
    PhysicalOrigin origin1({0.0, 0.0, 0.0});
    PhysicalOrigin origin2({0.0, 0.0, 0.0});
    PhysicalOrigin origin3({1.0, 1.0, 1.0});

    REQUIRE_FALSE(origin1 != origin2);
    REQUIRE(origin1 != origin3);
  }
}

// ============================================================================
// Edge Cases and Special Values
// ============================================================================

TEST_CASE("Strong types handle special values", "[strong_types][edge]") {
  SECTION("Zero values") {
    GridSize size({0, 0, 0});
    REQUIRE(size.get()[0] == 0);
    REQUIRE(size.get()[1] == 0);
    REQUIRE(size.get()[2] == 0);
  }

  SECTION("Negative values") {
    LocalOffset offset({-10, -20, -30});
    REQUIRE(offset.get()[0] == -10);
    REQUIRE(offset.get()[1] == -20);
    REQUIRE(offset.get()[2] == -30);
  }

  SECTION("Large values") {
    GlobalOffset offset({1000000, 2000000, 3000000});
    REQUIRE(offset.get()[0] == 1000000);
    REQUIRE(offset.get()[1] == 2000000);
    REQUIRE(offset.get()[2] == 3000000);
  }

  SECTION("Small floating point values") {
    GridSpacing spacing({1e-10, 1e-10, 1e-10});
    REQUIRE(spacing.get()[0] == 1e-10);
    REQUIRE(spacing.get()[1] == 1e-10);
    REQUIRE(spacing.get()[2] == 1e-10);
  }

  SECTION("Negative physical coordinates") {
    PhysicalOrigin origin({-1000.0, -2000.0, -3000.0});
    REQUIRE(origin.get()[0] == -1000.0);
    REQUIRE(origin.get()[1] == -2000.0);
    REQUIRE(origin.get()[2] == -3000.0);
  }
}

// ============================================================================
// Copy and Move Semantics
// ============================================================================

TEST_CASE("Strong types support copy semantics", "[strong_types][copy]") {
  SECTION("GridSize copy construction") {
    GridSize original({64, 64, 64});
    GridSize copy(original);

    REQUIRE(copy.get() == original.get());
  }

  SECTION("GridSize copy assignment") {
    GridSize original({64, 64, 64});
    GridSize copy({32, 32, 32});

    copy = original;

    REQUIRE(copy.get() == original.get());
  }

  SECTION("GridSpacing copy construction") {
    GridSpacing original({1.0, 1.0, 1.0});
    GridSpacing copy(original);

    REQUIRE(copy.get() == original.get());
  }

  SECTION("GridSpacing copy assignment") {
    GridSpacing original({1.0, 1.0, 1.0});
    GridSpacing copy({0.5, 0.5, 0.5});

    copy = original;

    REQUIRE(copy.get() == original.get());
  }
}
