// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/box3i.hpp>

using pfc::Box3i;
using pfc::for_each_index;

TEST_CASE("Single-cell box has volume 1", "[box3i][edge_case]") {
  Box3i box = Box3i::from_bounds({5, 5, 5}, {5, 5, 5});
  CHECK(box.count() == 1);
}

TEST_CASE("Single-cell box visits exactly one index via for_each_index", "[box3i][edge_case]") {
  Box3i box = Box3i::from_bounds({3, 3, 3}, {3, 3, 3});
  int call_count = 0;
  std::array<int, 3> last_visited;
  for_each_index(box, [&](std::array<int, 3> idx) {
    ++call_count;
    last_visited = idx;
  });
  CHECK(call_count == 1);
  CHECK(last_visited[0] == 3);
  CHECK(last_visited[1] == 3);
  CHECK(last_visited[2] == 3);
}

TEST_CASE("contains() returns true at low corner", "[box3i][edge_case]") {
  Box3i box = Box3i::from_bounds({2, 5, 8}, {7, 10, 13});
  CHECK(box.contains({2, 5, 8}) == true);
}

TEST_CASE("contains() returns true at high corner", "[box3i][edge_case]") {
  Box3i box = Box3i::from_bounds({0, 1, 2}, {7, 10, 13});
  CHECK(box.contains({0, 1, 2}) == true);
}

TEST_CASE("contains() returns false one step outside on each axis", "[box3i][edge_case]") {
  Box3i box = Box3i::from_bounds({5, 5, 5}, {10, 10, 10});
  CHECK(box.contains({4, 5, 5}) == false);  // low x - 1
  CHECK(box.contains({11, 5, 5}) == false);  // high x + 1
  CHECK(box.contains({5, 4, 5}) == false);  // low y - 1
  CHECK(box.contains({5, 11, 5}) == false);  // high y + 1
  CHECK(box.contains({5, 5, 4}) == false);  // low z - 1
  CHECK(box.contains({5, 5, 11}) == false);  // high z + 1
}

TEST_CASE("from_bounds with asymmetric extents yields correct sizes", "[box3i][edge_case]") {
  auto box = Box3i::from_bounds({0, 0, 0}, {2, 3, 1});
  CHECK(box.size[0] == 3);
  CHECK(box.size[1] == 4);
  CHECK(box.size[2] == 2);
}
