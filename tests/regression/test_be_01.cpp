// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_be_01.cpp
 * @brief Regression test for PE (periodicity) - M1: Domain-level periodicity validation
 *
 * This is the re-pointed version of the Pre-M0 PE regression test, now exercising
 * the Domain abstraction instead of World. The test validates that Domain correctly
 * handles per-axis periodicity, ensuring that periodic vs non-periodic axes are properly
 * distinguished and affect coordinate transformations appropriately.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/data/domain.hpp>

using namespace Catch::Matchers;
using pfc::Bool3;
using pfc::Domain;
using pfc::Int3;
using pfc::Real3;

// Domain namespace helpers
namespace domain = pfc::domain;

TEST_CASE("Domain regression test PE: per-axis periodicity affects spacing calculation",
          "[domain][regression][PE]") {
  // PE (audit 4.5): World's get_lower/upper_bounds ignored m_lower/m_upper for subdomains
  // causing them to report global origin. The fix was to use get_lower/get_upper(world).
  // In Domain architecture, this manifests as per-axis periodicity flags affecting
  // the spacing calculation: periodic axes use size division, non-periodic use size-1.

  SECTION("Periodic axis: dx = (upper - lower) / size") {
    // When an axis is periodic, the spacing is (upper - lower) / size
    const Domain d =
        domain::from_bounds({100, 100, 100}, {0.0, 0.0, 0.0}, {10.0, 10.0, 10.0},
                           {true, true, true});

    REQUIRE_THAT(domain::get_spacing(d, 0), WithinAbs(0.1, 1e-10));
    REQUIRE_THAT(domain::get_spacing(d, 1), WithinAbs(0.1, 1e-10));
    REQUIRE_THAT(domain::get_spacing(d, 2), WithinAbs(0.1, 1e-10));

    // Periodicity flags must be stored correctly
    REQUIRE(domain::is_periodic(d, 0) == true);
    REQUIRE(domain::is_periodic(d, 1) == true);
    REQUIRE(domain::is_periodic(d, 2) == true);
  }

  SECTION("Non-periodic axis: dx = (upper - lower) / (size - 1)") {
    // When an axis is non-periodic, the spacing is (upper - lower) / (size - 1)
    const Domain d =
        domain::from_bounds({100, 100, 100}, {0.0, 0.0, 0.0}, {10.0, 10.0, 10.0},
                           {false, false, false});

    double expected = 10.0 / 99.0;
    REQUIRE_THAT(domain::get_spacing(d, 0), WithinAbs(expected, 1e-10));
    REQUIRE_THAT(domain::get_spacing(d, 1), WithinAbs(expected, 1e-10));
    REQUIRE_THAT(domain::get_spacing(d, 2), WithinAbs(expected, 1e-10));

    // Periodicity flags must be stored correctly
    REQUIRE(domain::is_periodic(d, 0) == false);
    REQUIRE(domain::is_periodic(d, 1) == false);
    REQUIRE(domain::is_periodic(d, 2) == false);
  }

  SECTION("Mixed periodicity: each axis uses correct formula") {
    // Periodic in x and z, non-periodic in y
    const Domain d =
        domain::from_bounds({100, 100, 100}, {0.0, 0.0, 0.0}, {10.0, 20.0, 30.0},
                           {true, false, true});

    // x: periodic, so dx = 10/100 = 0.1
    REQUIRE_THAT(domain::get_spacing(d, 0), WithinAbs(0.1, 1e-10));
    // y: non-periodic, so dx = 20/99
    REQUIRE_THAT(domain::get_spacing(d, 1), WithinAbs(20.0 / 99.0, 1e-10));
    // z: periodic, so dx = 30/100 = 0.3
    REQUIRE_THAT(domain::get_spacing(d, 2), WithinAbs(0.3, 1e-10));

    // Stored periodicity flags must match the request {true, false, true}
    REQUIRE(domain::is_periodic(d, 0) == true);
    REQUIRE(domain::is_periodic(d, 1) == false);
    REQUIRE(domain::is_periodic(d, 2) == true);
  }

  SECTION("Domain periodicity flags are accessible and correct") {
    const Domain d1 = domain::create(pfc::GridSize({64, 32, 16}).to_vector3()); // Default: all periodic
    REQUIRE(domain::get_periodic(d1) == Bool3{true, true, true});

    const Domain d2 = domain::with_spacing({8, 8, 8}, {1.0, 1.0, 1.0},
                                            {false, true, false});
    REQUIRE(domain::is_periodic(d2, 0) == false);
    REQUIRE(domain::is_periodic(d2, 1) == true);
    REQUIRE(domain::is_periodic(d2, 2) == false);
    REQUIRE(domain::get_periodic(d2) == Bool3{false, true, false});
  }

  SECTION("Domain bounds are calculated correctly based on periodicity") {
    // Non-periodic: spacing = (upper-lower)/(size-1), so the far grid point sits
    // exactly on the upper bound.
    const Domain d =
        domain::from_bounds({100, 100, 100}, {0, 0, 0}, {9.9, 9.9, 9.9},
                           {false, false, false});

    REQUIRE(domain::get_lower_bounds(d) == Real3{0.0, 0.0, 0.0});

    const Real3 up = domain::get_upper_bounds(d);
    REQUIRE(up[0] == Catch::Approx(9.9));
    REQUIRE(up[1] == Catch::Approx(9.9));
    REQUIRE(up[2] == Catch::Approx(9.9));
  }
}
