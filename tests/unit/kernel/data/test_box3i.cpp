// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/fft/box3i.hpp>

using pfc::Box3i;

TEST_CASE("Box3i::from_bounds computes a consistent size", "[box3i][unit]") {
  const auto b = Box3i::from_bounds({0, 2, -3}, {3, 5, 0});
  REQUIRE(b.low == std::array<int, 3>{0, 2, -3});
  REQUIRE(b.high == std::array<int, 3>{3, 5, 0});
  REQUIRE(b.size == std::array<int, 3>{4, 4, 4});
  REQUIRE(b.is_consistent());
  REQUIRE(b.count() == 64);
}

TEST_CASE("Box3i::contains respects inclusive bounds", "[box3i][unit]") {
  const auto b = Box3i::from_bounds({0, 0, 0}, {2, 2, 2});
  REQUIRE(b.contains({0, 0, 0}));
  REQUIRE(b.contains({2, 2, 2}));
  REQUIRE_FALSE(b.contains({3, 0, 0}));
  REQUIRE_FALSE(b.contains({0, -1, 0}));
}

TEST_CASE("Box3i detects an inconsistent hand-built size", "[box3i][unit]") {
  Box3i bad{{0, 0, 0}, {3, 3, 3}, {2, 4, 4}}; // size[0] wrong
  REQUIRE_FALSE(bad.is_consistent());
}

TEST_CASE("fft::Box3i is the same type as pfc::Box3i", "[box3i][fft][unit]") {
  static_assert(std::is_same_v<pfc::fft::Box3i, pfc::Box3i>,
                "fft::Box3i must alias pfc::Box3i after M1");
  pfc::fft::Box3i b = Box3i::from_bounds({1, 1, 1}, {2, 2, 2});
  REQUIRE(b.count() == 8);
}
