// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/fft/box3i.hpp>

#include <sstream>
#include <vector>

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

TEST_CASE("Box3i stream operator", "[box3i][unit]") {
  pfc::Box3i box;
  box.low = {0, 1, 2};
  box.high = {3, 4, 5};
  box.size = {3, 3, 3};
  std::ostringstream oss;
  oss << box;
  std::string output = oss.str();
  REQUIRE(output.find("Box3i") != std::string::npos);
  REQUIRE(output.find("0") != std::string::npos);
  REQUIRE(output.find("5") != std::string::npos);
}

TEST_CASE("for_each_index visits all indices in x-fastest order", "[box3i][unit]") {
  Box3i box = Box3i::from_bounds({0, 0, 0}, {2, 1, 1});  // small non-cubic: 3x2x2 = 12 points

  std::vector<std::array<int, 3>> visited;
  pfc::for_each_index(box, [&](std::array<int, 3> idx) { visited.push_back(idx); });

  SECTION("visit count matches count()") {
    REQUIRE(visited.size() == static_cast<size_t>(box.count()));
  }

  SECTION("first index is low, last is high") {
    REQUIRE(visited.front() == box.low);
    REQUIRE(visited.back() == box.high);
  }

  SECTION("x varies fastest") {
    REQUIRE(visited[1] != visited[0]);
    REQUIRE(visited[1][0] == visited[0][0] + 1);  // x increment
    REQUIRE(visited[1][1] == visited[0][1]);       // y unchanged
    REQUIRE(visited[1][2] == visited[0][2]);       // z unchanged
  }
}
