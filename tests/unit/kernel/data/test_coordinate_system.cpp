// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/discrete_field.hpp>

using namespace Catch::Matchers;
using namespace pfc;
using namespace pfc::domain;
using namespace pfc::types;

// ============================================================================
// CSYS coordinate transformation tests
// ============================================================================

TEST_CASE("Domain::to_indices rounds correctly", "[domain][coordinate_system]") {
  // This test validates that coordinate-to-index conversion rounds to the
  // nearest integer instead of truncating, matching the documented contract
  // and DiscreteField behavior.

  SECTION("rounding at cell midpoints") {
    const Domain d = create(GridSize({100, 100, 100}), PhysicalOrigin({0.0, 0.0, 0.0}), GridSpacing({1.0, 1.0, 1.0}));

    // Test positive coordinates: epsilon values < 0.5 should stay at center,
    // >= 0.5 should round up
    CHECK(to_indices(d, {2.25, 2.25, 2.25}) == Int3{2, 2, 2});  // ε=0.25, stays
    CHECK(to_indices(d, {2.49, 2.49, 2.49}) == Int3{2, 2, 2});  // ε=0.49, stays
    CHECK(to_indices(d, {2.51, 2.51, 2.51}) == Int3{3, 3, 3});  // ε=0.51, rounds up
    CHECK(to_indices(d, {2.75, 2.75, 2.75}) == Int3{3, 3, 3});  // ε=0.75, rounds up

    // Test negative coordinates: also round correctly
    CHECK(to_indices(d, {-2.25, -2.25, -2.25}) == Int3{-2, -2, -2});  // ε=0.25, stays
    CHECK(to_indices(d, {-2.49, -2.49, -2.49}) == Int3{-2, -2, -2});  // ε=0.49, stays
    CHECK(to_indices(d, {-2.51, -2.51, -2.51}) == Int3{-3, -3, -3});  // ε=0.51, rounds down
    CHECK(to_indices(d, {-2.75, -2.75, -2.75}) == Int3{-3, -3, -3});  // ε=0.75, rounds down

    // Test half-way values: round away from zero
    CHECK(to_indices(d, {2.5, 2.5, 2.5}) == Int3{3, 3, 3});    // Positive: rounds up
    CHECK(to_indices(d, {-2.5, -2.5, -2.5}) == Int3{-3, -3, -3}); // Negative: rounds down
  }

  SECTION("rounding with non-unit spacing") {
    const Domain d = with_spacing({100, 100, 100}, {0.5, 0.5, 0.5});

    // With spacing 0.5, coordinates 0.25 and 0.26 map to index 0 and 1 respectively
    CHECK(to_indices(d, {0.24, 0.0, 0.0})[0] == 0);  // 0.24/0.5 = 0.48 → 0
    CHECK(to_indices(d, {0.26, 0.0, 0.0})[0] == 1);  // 0.26/0.5 = 0.52 → 1

    // Check various coordinate values with spacing 0.5
    CHECK(to_indices(d, {1.24, 0.0, 0.0})[0] == 2);  // 1.24/0.5 = 2.48 → 2
    CHECK(to_indices(d, {1.26, 0.0, 0.0})[0] == 3);  // 1.26/0.5 = 2.52 → 3

    // Test with spacing other than 0.5
    const Domain d2 = with_spacing({100, 100, 100}, {0.1, 0.2, 0.5});
    CHECK(to_indices(d2, {1.06, 0.0, 0.0}) == Int3{11, 0, 0});   // 1.06/0.1 = 10.6 → 11
    CHECK(to_indices(d2, {0.0, 0.76, 0.0}) == Int3{0, 4, 0});    // 0.76/0.2 = 3.8 → 4
    CHECK(to_indices(d2, {0.0, 0.0, 0.75}) == Int3{0, 0, 2});    // 0.75/0.5 = 1.5 → 2
  }

  SECTION("rounding with non-zero origin") {
    const Domain d = create(GridSize({100, 100, 100}), PhysicalOrigin({1.5, 1.5, 1.5}), GridSpacing({1.0, 1.0, 1.0}));

    // Coordinate mapping: index = round((coord - origin) / spacing)
    // For origin=1.5, spacing=1.0:
    // coord=3.74: (3.74-1.5)/1.0 = 2.24 → 2
    // coord=3.76: (3.76-1.5)/1.0 = 2.26 → 2 (rounds to 2, not 3)
    // coord=0.0: (0.0-1.5)/1.0 = -1.5 → -2 (rounds to -2)
    CHECK(to_indices(d, {3.74, 1.5, 1.5}) == Int3{2, 0, 0});
    CHECK(to_indices(d, {3.76, 1.5, 1.5}) == Int3{2, 0, 0});

    // Test negative origin
    const Domain d2 = create(GridSize({100, 100, 100}), PhysicalOrigin({-3.0, -2.0, -1.0}), GridSpacing({1.0, 1.0, 1.0}));
    CHECK(to_indices(d2, {-0.74, -2.0, -1.0}) == Int3{2, 0, 0});  // (-0.74+3.0)/1.0 = 2.26 → 2
    CHECK(to_indices(d2, {-0.76, -2.0, -1.0}) == Int3{2, 0, 0});  // (-0.76+3.0)/1.0 = 2.24 → 2
  }

  SECTION("rounding consistency round-trip") {
    const Domain d = create(Int3{100, 100, 100});

    // Start with integer indices, convert to coords, then back to indices
    const Int3 probe = {50, 25, 75};
    const Real3 coords = to_coords(d, probe);

    // Nudge by <half a cell in both directions: must round back to the same index
    const Real3 epsilon{0.49, 0.49, 0.49};
    const Real3 plus{coords[0] + epsilon[0], coords[1] + epsilon[1], coords[2] + epsilon[2]};
    const Real3 minus{coords[0] - epsilon[0], coords[1] - epsilon[1], coords[2] - epsilon[2]};

    CHECK(to_indices(d, plus) == probe);
    CHECK(to_indices(d, minus) == probe);
  }

  SECTION("rounding at domain boundaries") {
    const Domain d = with_spacing({10, 10, 10}, {1.0, 1.0, 1.0});

    // Near lower bound: 0.49 rounds to 0, 0.51 rounds to 1
    CHECK(to_indices(d, {0.49, 0.49, 0.49}) == Int3{0, 0, 0});
    CHECK(to_indices(d, {0.51, 0.51, 0.51}) == Int3{1, 1, 1});

    // Near upper bound: 8.49 rounds to 8, 8.51 rounds to 9
    CHECK(to_indices(d, {8.49, 8.49, 8.49}) == Int3{8, 8, 8});
    CHECK(to_indices(d, {8.51, 8.51, 8.51}) == Int3{9, 9, 9});

    // At boundary: 9.5 rounds to 10 (out of bounds for size 10)
    CHECK(to_indices(d, {9.5, 9.5, 9.5}) == Int3{10, 10, 10});
  }
}

TEST_CASE("DiscreteField coordinate transformations round correctly",
          "[discrete_field][coordinate_system]") {
  SECTION("DiscreteField uses rounding for coordinate-to-index conversion") {
    // Create a 3D field with unit spacing
    DiscreteField<double, 3> field({64, 64, 64}, {0, 0, 0}, {0.0, 0.0, 0.0},
                                    {1.0, 1.0, 1.0});

    // Test rounding behavior
    CHECK(field.map_coordinates_to_indices({10.25, 20.25, 30.25}) ==
          std::array<int, 3>{10, 20, 30});  // ε=0.25, stays
    CHECK(field.map_coordinates_to_indices({10.49, 20.49, 30.49}) ==
          std::array<int, 3>{10, 20, 30});  // ε=0.49, stays
    CHECK(field.map_coordinates_to_indices({10.51, 20.51, 30.51}) ==
          std::array<int, 3>{11, 21, 31});  // ε=0.51, rounds up
    CHECK(field.map_coordinates_to_indices({10.75, 20.75, 30.75}) ==
          std::array<int, 3>{11, 21, 31});  // ε=0.75, rounds up
  }

  SECTION("DiscreteField round-trip stability") {
    DiscreteField<double, 3> field({64, 64, 64}, {0, 0, 0}, {0.0, 0.0, 0.0},
                                    {1.0, 1.0, 1.0});

    // Start with integer indices, convert to coords, then back to indices
    const std::array<int, 3> probe = {32, 16, 48};
    const auto coords = field.map_indices_to_coordinates(probe);

    // Nudge by <half a cell and verify round-trip
    const std::array<double, 3> epsilon{0.49, 0.49, 0.49};
    std::array<double, 3> plus = coords;
    std::array<double, 3> minus = coords;
    for (int i = 0; i < 3; ++i) {
      plus[i] += epsilon[i];
      minus[i] -= epsilon[i];
    }

    CHECK(field.map_coordinates_to_indices(plus) == probe);
    CHECK(field.map_coordinates_to_indices(minus) == probe);
  }

  SECTION("DiscreteField with non-unit spacing") {
    DiscreteField<double, 3> field({100, 100, 100}, {0, 0, 0}, {0.0, 0.0, 0.0},
                                    {0.5, 0.5, 0.5});

    // With spacing 0.5, coordinates 0.25 and 0.26 map to index 0 and 1 respectively
    CHECK(field.map_coordinates_to_indices({0.24, 0.0, 0.0})[0] == 0);
    CHECK(field.map_coordinates_to_indices({0.26, 0.0, 0.0})[0] == 1);
    CHECK(field.map_coordinates_to_indices({1.24, 1.0, 1.0})[0] == 2);
    CHECK(field.map_coordinates_to_indices({1.26, 1.0, 1.0})[0] == 3);
  }

  SECTION("DiscreteField with non-zero origin") {
    DiscreteField<double, 3> field({100, 100, 100}, {0, 0, 0}, {1.5, 1.5, 1.5},
                                    {1.0, 1.0, 1.0});

    // Coordinate mapping should handle origin correctly
    // origin=1.5: coord=3.74 -> (3.74-1.5)/1.0 = 2.24 -> round to 2
    //            coord=3.76 -> (3.76-1.5)/1.0 = 2.26 -> round to 2
    CHECK(field.map_coordinates_to_indices({3.74, 1.5, 1.5}) == std::array<int, 3>{2, 0, 0});
    CHECK(field.map_coordinates_to_indices({3.76, 1.5, 1.5}) == std::array<int, 3>{2, 0, 0});
  }
}

TEST_CASE("Domain and DiscreteField coordinate transformations agree",
          "[domain][discrete_field][coordinate_system][agreement]") {
  SECTION("Both use rounding consistently") {
    const Domain d = create(Int3{64, 64, 64});
    DiscreteField<double, 3> field({64, 64, 64}, {0, 0, 0}, {0.0, 0.0, 0.0},
                                    {1.0, 1.0, 1.0});

    // Test various coordinates that should round the same way
    const std::vector<Real3> test_coords{
        {5.25, 10.25, 15.25},  // ε=0.25, should stay
        {5.49, 10.49, 15.49},  // ε=0.49, should stay
        {5.51, 10.51, 15.51},  // ε=0.51, should round up
        {5.75, 10.75, 15.75},  // ε=0.75, should round up
        {10.5, 20.5, 30.5},    // half-way, round up
        {-5.25, -10.25, -15.25}, // negative ε=0.25, should stay
        {-5.51, -10.51, -15.51}  // negative ε=0.51, should round down
    };

    for (const auto &coords : test_coords) {
      const Int3 idx_from_domain = to_indices(d, coords);
      const std::array<int, 3> idx_from_field = field.map_coordinates_to_indices(coords);

      CHECK(idx_from_domain[0] == idx_from_field[0]);
      CHECK(idx_from_domain[1] == idx_from_field[1]);
      CHECK(idx_from_domain[2] == idx_from_field[2]);
    }
  }

  SECTION("Agreement with non-unit spacing") {
    const Domain d = with_spacing(Int3{64, 64, 64}, Real3{0.5, 0.5, 0.5});
    DiscreteField<double, 3> field({64, 64, 64}, {0, 0, 0}, {0.0, 0.0, 0.0},
                                    {0.5, 0.5, 0.5});

    const std::vector<Real3> test_coords{
        {1.24, 2.24, 3.24},  // rounds down
        {1.26, 2.26, 3.26},  // rounds up
        {10.5, 20.5, 30.5}    // half-way, round up
    };

    for (const auto &coords : test_coords) {
      const Int3 idx_from_domain = to_indices(d, coords);
      const std::array<int, 3> idx_from_field = field.map_coordinates_to_indices(coords);

      CHECK(idx_from_domain[0] == idx_from_field[0]);
      CHECK(idx_from_domain[1] == idx_from_field[1]);
      CHECK(idx_from_domain[2] == idx_from_field[2]);
    }
  }

  SECTION("Agreement with non-zero origin") {
    const Domain d = create(GridSize({64, 64, 64}), PhysicalOrigin({1.5, 1.5, 1.5}), GridSpacing({1.0, 1.0, 1.0}));
    DiscreteField<double, 3> field({64, 64, 64}, {0, 0, 0}, {1.5, 1.5, 1.5},
                                    {1.0, 1.0, 1.0});

    const std::vector<Real3> test_coords{
        {5.0, 10.0, 15.0},   // exact
        {5.24, 10.24, 15.24}, // rounds down
        {5.26, 10.26, 15.26}  // rounds up
    };

    for (const auto &coords : test_coords) {
      const Int3 idx_from_domain = to_indices(d, coords);
      const std::array<int, 3> idx_from_field = field.map_coordinates_to_indices(coords);

      CHECK(idx_from_domain[0] == idx_from_field[0]);
      CHECK(idx_from_domain[1] == idx_from_field[1]);
      CHECK(idx_from_domain[2] == idx_from_field[2]);
    }
  }
}