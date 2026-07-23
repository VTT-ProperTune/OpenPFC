// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/world.hpp> // cross-check numerical parity with World

using pfc::Bool3;
using pfc::Box3i;
using pfc::Domain;
using pfc::Int3;
using pfc::Real3;
namespace domain = pfc::domain;

TEST_CASE("Domain::create defaults: unit spacing, zero origin, periodic",
          "[domain][unit]") {
  const Domain d = domain::create({64, 32, 16});
  REQUIRE(domain::get_size(d) == Int3{64, 32, 16});
  REQUIRE(domain::get_spacing(d) == Real3{1.0, 1.0, 1.0});
  REQUIRE(domain::get_origin(d) == Real3{0.0, 0.0, 0.0});
  REQUIRE(domain::get_periodic(d) == Bool3{true, true, true});
  REQUIRE(domain::get_total_size(d) == 64u * 32u * 16u);
}

TEST_CASE("Domain consumes per-axis periodicity", "[domain][unit]") {
  const Domain d =
      domain::with_spacing({8, 8, 8}, {1.0, 1.0, 1.0}, {false, true, false});
  REQUIRE(domain::is_periodic(d, 0) == false);
  REQUIRE(domain::is_periodic(d, 1) == true);
  REQUIRE(domain::is_periodic(d, 2) == false);
}

TEST_CASE("Domain rejects non-positive size/spacing", "[domain][unit]") {
  REQUIRE_THROWS_AS(domain::create(pfc::GridSize({0, 4, 4}),
                                   pfc::PhysicalOrigin({0, 0, 0}),
                                   pfc::GridSpacing({1, 1, 1})),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(domain::with_spacing({4, 4, 4}, {1.0, -1.0, 1.0}),
                    std::invalid_argument);
}

TEST_CASE("Domain::index_box is the global [0, size-1] box", "[domain][unit]") {
  const Domain d = domain::create({10, 20, 30});
  const Box3i b = domain::index_box(d);
  REQUIRE(b == Box3i::from_bounds({0, 0, 0}, {9, 19, 29}));
  REQUIRE(static_cast<size_t>(b.count()) == domain::get_total_size(d));
}

TEST_CASE("Domain coordinate round-trip uses nearest-grid rounding",
          "[domain][unit]") {
  const Domain d =
      domain::from_bounds({100, 100, 100}, {-5.0, -5.0, 0.0}, {5.0, 5.0, 10.0});
  const Int3 probe{37, 12, 88};
  const Real3 x = domain::to_coords(d, probe);
  // Nudge by <half a cell in both directions: must round back to the same index.
  Real3 dx = domain::get_spacing(d);
  Real3 plus{x[0] + 0.49 * dx[0], x[1] + 0.49 * dx[1], x[2] + 0.49 * dx[2]};
  Real3 minus{x[0] - 0.49 * dx[0], x[1] - 0.49 * dx[1], x[2] - 0.49 * dx[2]};
  REQUIRE(domain::to_indices(d, plus) == probe);
  REQUIRE(domain::to_indices(d, minus) == probe);
}

TEST_CASE("Domain dimensionality / bounds / volume", "[domain][unit]") {
  REQUIRE(domain::dimensionality(domain::create({100, 1, 1})) == 1);
  REQUIRE(domain::dimensionality(domain::create({64, 64, 1})) == 2);
  REQUIRE(domain::dimensionality(domain::create({32, 32, 32})) == 3);
  REQUIRE(domain::dimensionality(domain::create({1, 1, 1})) == 0);

  // Non-periodic: spacing = (upper-lower)/(size-1), so the far grid point sits
  // exactly on the upper bound.
  const Domain d = domain::from_bounds({100, 100, 100}, {0, 0, 0}, {9.9, 9.9, 9.9},
                                       {false, false, false});
  REQUIRE(domain::get_lower_bounds(d) == Real3{0.0, 0.0, 0.0});
  const Real3 up = domain::get_upper_bounds(d);
  REQUIRE(up[0] == Catch::Approx(9.9));
  REQUIRE(domain::physical_volume(domain::with_spacing(
              {10, 10, 10}, {0.1, 0.1, 0.1})) == Catch::Approx(0.001 * 1000));
}

// The migration in M1.3–M1.5 is only safe if Domain reproduces World's numerics
// bit-for-bit on the shared surface. Pin that here against the live World code.
TEST_CASE("Domain matches World numerically on the shared surface",
          "[domain][world][unit]") {
  const Int3 size{100, 80, 60};
  const Real3 lower{-5.0, -2.0, 0.0};
  const Real3 upper{5.0, 6.0, 12.0};

  const Domain d = domain::from_bounds(size, lower, upper);
  const auto w = pfc::world::from_bounds(size, lower, upper);

  REQUIRE(domain::get_size(d) == pfc::world::get_size(w));
  REQUIRE(domain::get_spacing(d) == pfc::world::get_spacing(w));
  REQUIRE(domain::get_total_size(d) == pfc::world::get_total_size(w));
  REQUIRE(domain::dimensionality(d) == pfc::world::dimensionality(w));
  REQUIRE(domain::physical_volume(d) == pfc::world::physical_volume(w));
  REQUIRE(domain::get_lower_bounds(d) == pfc::world::get_lower_bounds(w));
  REQUIRE(domain::get_upper_bounds(d) == pfc::world::get_upper_bounds(w));

  for (const Int3 idx : {Int3{0, 0, 0}, Int3{50, 40, 30}, Int3{99, 79, 59}}) {
    REQUIRE(domain::to_coords(d, idx) == pfc::world::to_coords(w, idx));
    const Real3 x = domain::to_coords(d, idx);
    REQUIRE(domain::to_indices(d, x) == pfc::world::to_indices(w, x));
  }
}
