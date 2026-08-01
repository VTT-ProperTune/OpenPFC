// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_be_02.cpp
 * @brief Regression test for PF (coordinate rounding) - M1: Domain-level coordinate rounding validation
 *
 * This is the re-pointed version of the Pre-M0 PF regression test, now exercising
 * the Domain abstraction instead of World. The test validates that Domain correctly
 * uses nearest-grid rounding when converting physical coordinates to indices, ensuring
 * the documented contract (round to nearest integer) is honored.
 */

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/domain.hpp>

using pfc::Domain;
using pfc::GridSize;
using pfc::GridSpacing;
using pfc::Int3;
using pfc::PhysicalOrigin;
using pfc::Real3;

// Domain namespace helpers
namespace domain = pfc::domain;

TEST_CASE("Domain regression test PF: coordinate rounding uses nearest-grid rounding",
          "[domain][regression][PF]") {
  // PF (audit 4.6): World's to_index truncated while to_indices documented (and
  // coordinate systems) rounding. The fix standardized on std::lround.
  // This test validates that Domain::to_indices uses proper rounding to nearest
  // integer, not truncation.

  const Int3 dimensions = {100, 100, 100};
  const Real3 origin = {0.0, 0.0, 0.0};
  const Real3 spacing = {1.0, 1.0, 1.0};
  const Domain d = domain::create(GridSize(dimensions), PhysicalOrigin(origin), GridSpacing(spacing));

  SECTION("Standard nearest-grid rounding behavior") {
    // 10.6 must round up to 11 (truncation would wrongly give 10);
    // 20.4 rounds down to 20; 30.5 rounds half away from zero to 31.
    const Real3 coords = {10.6, 20.4, 30.5};
    const Int3 indices = domain::to_indices(d, coords);
    REQUIRE(indices == Int3{11, 20, 31});
  }

  SECTION("Round-trip stability: coordinates round-trip correctly") {
    // Start with integer indices, convert to coords, then back to indices
    const Int3 probe = {37, 12, 88};
    const Real3 coords = domain::to_coords(d, probe);

    // Nudge by <half a cell in both directions: must round back to the same index
    const Real3 dx = domain::get_spacing(d);
    const Real3 plus{coords[0] + 0.49 * dx[0], coords[1] + 0.49 * dx[1],
                     coords[2] + 0.49 * dx[2]};
    const Real3 minus{coords[0] - 0.49 * dx[0], coords[1] - 0.49 * dx[1],
                      coords[2] - 0.49 * dx[2]};

    REQUIRE(domain::to_indices(d, plus) == probe);
    REQUIRE(domain::to_indices(d, minus) == probe);
  }

  SECTION("Negative coordinates round correctly") {
    // Test that negative coordinates also round properly (away from zero on halves)
    const Real3 coords = {-10.6, -20.4, -30.5};
    const Int3 indices = domain::to_indices(d, coords);
    // -10.6 rounds to -11, -20.4 rounds to -20, -30.5 rounds to -31
    REQUIRE(indices == Int3{-11, -20, -31});
  }

  SECTION("Half-way values round consistently") {
    // Test that half-way values (x.5) round away from zero consistently
    const Real3 coords = {10.5, -10.5, 0.5};
    const Int3 indices = domain::to_indices(d, coords);
    // 10.5 rounds to 11, -10.5 rounds to -11, 0.5 rounds to 1
    REQUIRE(indices == Int3{11, -11, 1});
  }

  SECTION("Rounding respects spacing (non-unit spacing)") {
    // Test with non-unit spacing to ensure rounding works with different scales
    const Domain d2 = domain::with_spacing({100, 100, 100}, {0.1, 0.2, 0.5});

    // At spacing 0.1, 1.06 should map to index 11 (coords / 0.1 = 10.6 -> round to 11)
    const Real3 coords1 = {1.06, 0.0, 0.0};
    const Int3 indices1 = domain::to_indices(d2, coords1);
    REQUIRE(indices1[0] == 11);

    // At spacing 0.2, 0.76 should map to index 4 (0.76 / 0.2 = 3.8 -> round to 4)
    const Real3 coords2 = {0.0, 0.76, 0.0};
    const Int3 indices2 = domain::to_indices(d2, coords2);
    REQUIRE(indices2[1] == 4);

    // At spacing 0.5, 0.75 should map to index 2 (0.75 / 0.5 = 1.5 -> round to 2)
    const Real3 coords3 = {0.0, 0.0, 0.75};
    const Int3 indices3 = domain::to_indices(d2, coords3);
    REQUIRE(indices3[2] == 2);
  }

  SECTION("Bounded coordinates at domain edges round correctly") {
    // Test coordinates near domain boundaries
    const Domain d3 = domain::with_spacing({10, 10, 10}, {1.0, 1.0, 1.0});

    // Near lower bound
    const Real3 near_lower = {0.4, 0.4, 0.4};
    const Int3 indices_lower = domain::to_indices(d3, near_lower);
    REQUIRE(indices_lower == Int3{0, 0, 0});

    // Near upper bound - 9.6 should round to 10, but domain is [0,9]
    const Real3 near_upper = {9.6, 9.6, 9.6};
    const Int3 indices_upper = domain::to_indices(d3, near_upper);
    // 9.6 rounds to 10, which is out of bounds for a domain of size 10
    REQUIRE(indices_upper == Int3{10, 10, 10});
  }
}
