// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/data/domain.hpp>

#include <catch2/catch_test_macros.hpp>
#include <numbers>

using namespace pfc;

TEST_CASE("DiffOp - differentiate in all directions", "[diffop][unit]") {
  // Setup
  const Real3 lo{-std::numbers::pi, -std::numbers::pi, -std::numbers::pi};
  const Real3 hi{std::numbers::pi, std::numbers::pi, std::numbers::pi};
  const Int3 size{128, 128, 128};
  auto domain = pfc::domain::from_bounds(size, lo, hi);

  // Validate domain parameters
  SECTION("Domain creation from bounds") {
    REQUIRE(domain::get_size(domain) == Int3{128, 128, 128});
    REQUIRE(domain::get_origin(domain)[0] < 0.0);
    REQUIRE(domain::get_origin(domain)[1] < 0.0);
    REQUIRE(domain::get_origin(domain)[2] < 0.0);
    REQUIRE(domain::get_upper_bounds(domain)[0] > 0.0);
    REQUIRE(domain::get_upper_bounds(domain)[1] > 0.0);
    REQUIRE(domain::get_upper_bounds(domain)[2] > 0.0);
  }

  SECTION("Domain spacing is positive") {
    REQUIRE(domain::get_spacing(domain)[0] > 0.0);
    REQUIRE(domain::get_spacing(domain)[1] > 0.0);
    REQUIRE(domain::get_spacing(domain)[2] > 0.0);
  }

  SECTION("Domain is periodic") {
    REQUIRE(domain::is_periodic(domain, 0));
    REQUIRE(domain::is_periodic(domain, 1));
    REQUIRE(domain::is_periodic(domain, 2));
  }

  SECTION("Domain total size") {
    const size_t expected_total = static_cast<size_t>(128) * 128 * 128;
    REQUIRE(domain::get_total_size(domain) == expected_total);
  }
}
