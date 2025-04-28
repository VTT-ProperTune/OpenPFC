// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/core/box3d.hpp"
#include <catch2/catch_test_macros.hpp>

using namespace pfc;

TEST_CASE("Box3D construction and basic properties", "[box3d]") {
  Box3D::Int3 lower = {0, 0, 0};
  Box3D::Int3 upper = {9, 19, 29};
  Box3D box(lower, upper);

  REQUIRE(box.lower() == lower);
  REQUIRE(box.upper() == upper);
  REQUIRE(box.size() == Box3D::Int3{10, 20, 30});
  REQUIRE(box.total_size() == 10 * 20 * 30);
}

TEST_CASE("Box3D contains() method", "[box3d]") {
  Box3D box({0, 0, 0}, {9, 9, 9});

  REQUIRE(box.contains({0, 0, 0}));
  REQUIRE(box.contains({5, 5, 5}));
  REQUIRE(box.contains({9, 9, 9}));

  REQUIRE_FALSE(box.contains({-1, 5, 5}));
  REQUIRE_FALSE(box.contains({5, -1, 5}));
  REQUIRE_FALSE(box.contains({5, 5, -1}));
  REQUIRE_FALSE(box.contains({10, 5, 5}));
}

TEST_CASE("Box3D equality and inequality", "[box3d]") {
  Box3D a({0, 0, 0}, {5, 5, 5});
  Box3D b({0, 0, 0}, {5, 5, 5});
  Box3D c({1, 0, 0}, {5, 5, 5});

  REQUIRE(a == b);
  REQUIRE(a != c);
}

TEST_CASE("Box3D throws on invalid construction", "[box3d]") {
  REQUIRE_THROWS_AS(Box3D({5, 0, 0}, {4, 0, 0}), std::invalid_argument);
  REQUIRE_THROWS_AS(Box3D({0, 6, 0}, {0, 5, 0}), std::invalid_argument);
  REQUIRE_THROWS_AS(Box3D({0, 0, 7}, {0, 0, 6}), std::invalid_argument);
}
