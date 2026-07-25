// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_strong_types.cpp
 * @brief Comprehensive tests for strong type aliases
 *
 * Tests verify:
 * - Zero-cost abstraction (same size as underlying types)
 * - Explicit conversions (from_vector3, to_vector3) and implicit conversions
 * - Type safety (different strong types are distinct)
 * - Construction patterns
 * - Comparison operators
 * - Standard layout and trivial copyability
 */

#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <type_traits>

using namespace pfc;

// ============================================================================
// Zero-Cost Abstraction Tests
// ============================================================================

TEST_CASE("Strong types are zero-cost abstractions", "[strong_types][performance]") {
  SECTION("GridSize has same size as Int3") {
    REQUIRE(sizeof(GridSize) == sizeof(Int3));
  }

  SECTION("GridSpacing has same size as Real3") {
    REQUIRE(sizeof(GridSpacing) == sizeof(Real3));
  }

  SECTION("PhysicalOrigin has same size as Real3") {
    REQUIRE(sizeof(PhysicalOrigin) == sizeof(Real3));
  }
}

TEST_CASE("Strong types are trivially copyable", "[strong_types][performance]") {
  SECTION("GridSize is trivially copyable") {
    REQUIRE(std::is_trivially_copyable_v<GridSize>);
  }

  SECTION("GridSpacing is trivially copyable") {
    REQUIRE(std::is_trivially_copyable_v<GridSpacing>);
  }

  SECTION("PhysicalOrigin is trivially copyable") {
    REQUIRE(std::is_trivially_copyable_v<PhysicalOrigin>);
  }
}

TEST_CASE("Strong types have standard layout", "[strong_types][performance]") {
  SECTION("GridSize has standard layout") {
    REQUIRE(std::is_standard_layout_v<GridSize>);
  }

  SECTION("GridSpacing has standard layout") {
    REQUIRE(std::is_standard_layout_v<GridSpacing>);
  }

  SECTION("PhysicalOrigin has standard layout") {
    REQUIRE(std::is_standard_layout_v<PhysicalOrigin>);
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

  SECTION("Explicit factory method from_vector3") {
    Int3 raw = {100, 100, 100};
    GridSize size = GridSize::from_vector3(raw);

    REQUIRE(size.get()[0] == 100);
    REQUIRE(size.get()[1] == 100);
    REQUIRE(size.get()[2] == 100);
  }

  SECTION("Explicit conversion method to_vector3") {
    GridSize size({100, 100, 100});
    Int3 converted = size.to_vector3();

    REQUIRE(converted[0] == 100);
    REQUIRE(converted[1] == 100);
    REQUIRE(converted[2] == 100);
  }

  SECTION("Explicit conversion back to Int3 works") {
    GridSize size({100, 100, 100});
    Int3 converted = size.to_vector3();  // Explicit conversion

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

  SECTION("Explicit factory method from_vector3") {
    Real3 raw = {0.25, 0.25, 0.25};
    GridSpacing spacing = GridSpacing::from_vector3(raw);

    REQUIRE(spacing.get()[0] == 0.25);
    REQUIRE(spacing.get()[1] == 0.25);
    REQUIRE(spacing.get()[2] == 0.25);
  }

  SECTION("Explicit conversion method to_vector3") {
    GridSpacing spacing({0.25, 0.25, 0.25});
    Real3 converted = spacing.to_vector3();

    REQUIRE(converted[0] == 0.25);
    REQUIRE(converted[1] == 0.25);
    REQUIRE(converted[2] == 0.25);
  }

  SECTION("Explicit conversion back to Real3 works") {
    GridSpacing spacing({0.25, 0.25, 0.25});
    Real3 converted = spacing.to_vector3();  // Explicit conversion

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

  SECTION("Explicit conversion back to Real3 works") {
    PhysicalOrigin origin({5.5, 10.5, 15.5});
    Real3 converted = origin.to_vector3();

    REQUIRE(converted[0] == 5.5);
    REQUIRE(converted[1] == 10.5);
    REQUIRE(converted[2] == 15.5);
  }
}

// ============================================================================
// Type Safety Tests
// ============================================================================

TEST_CASE("Strong types are distinct types", "[strong_types][safety]") {
  SECTION("GridSize and GridSpacing are different types") {
    REQUIRE_FALSE(std::is_same_v<GridSize, GridSpacing>);
  }

  SECTION("GridSize and PhysicalOrigin are different types") {
    REQUIRE_FALSE(std::is_same_v<GridSize, PhysicalOrigin>);
  }

  SECTION("GridSpacing and PhysicalOrigin are different types") {
    REQUIRE_FALSE(std::is_same_v<GridSpacing, PhysicalOrigin>);
  }
}

TEST_CASE("Strong types cannot be implicitly assigned to each other",
          "[strong_types][safety]") {
  SECTION("Cannot assign GridSize to GridSpacing") {
    // This test verifies that the types are distinct at compile time
    // The actual prevention happens at compile time, so we just verify the types are
    // different
    GridSize size({64, 64, 64});
    GridSpacing spacing({1.0, 1.0, 1.0});

    // These are different types
    REQUIRE_FALSE(std::is_same_v<decltype(size), decltype(spacing)>);
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
