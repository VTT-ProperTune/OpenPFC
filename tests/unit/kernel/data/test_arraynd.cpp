// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/data/array.hpp>

using namespace Catch::Matchers;
using namespace pfc;

TEST_CASE("array1d") {
  // array with dimension 5, offset 2
  Array<int, 1> arr({5}, {2});

  SECTION("Test setting and accessing elements using linear indexing") {
    arr[0] = 1;
    arr[1] = 2;
    REQUIRE(arr[0] == 1);
    REQUIRE(arr[1] == 2);
    REQUIRE(arr.get_data()[0] == 1);
    REQUIRE(arr.get_data()[1] == 2);
  }

  SECTION("Test setting and accessing elements using custom indexing") {
    arr({2}) = 1;
    arr({3}) = 2;
    REQUIRE(arr[0] == 1);
    REQUIRE(arr[1] == 2);
  }

  SECTION("Test in-bounds") {
    // 0 1 2 3 4 5 6
    //     ^
    //     0 1 2 3 4
    REQUIRE_FALSE(arr.inbounds({1}));
    REQUIRE(arr.inbounds({2}));
    REQUIRE(arr.inbounds({6}));
    REQUIRE_FALSE(arr.inbounds({7}));
  }

  SECTION("Test get_size()") { REQUIRE(arr.get_size()[0] == 5); }

  SECTION("Test apply()") {
    auto func = [](const std::array<int, 1> &indices) -> int {
      return 2 * indices[0];
    };
    arr.apply(func);
    REQUIRE(arr[0] == 4);
    REQUIRE(arr[1] == 6);
    REQUIRE(arr[2] == 8);
    REQUIRE(arr[3] == 10);
    REQUIRE(arr[4] == 12);
  }
}

TEST_CASE("array2d") {
  Array<int, 2> arr({2, 3}, {1, 2});
  /**
   *    2  3  4
   * 1  x  x  x
   * 2  x  x  x
   */

  SECTION("Test setting and accessing elements using custom / linear indexing") {
    arr({1, 2}) = 1;
    arr({2, 2}) = 2;
    arr({1, 3}) = 3;
    arr({2, 3}) = 4;
    arr({1, 4}) = 5;
    arr({2, 4}) = 6;
    bool values_match = true;
    for (int i = 0; i < 6; i++) values_match &= arr[i] == i + 1;
    REQUIRE(values_match);
  }

  SECTION("Test apply()") {
    auto func = [](const std::array<int, 2> &indices) -> int {
      return indices[0] + indices[1];
    };
    arr.apply(func);
    REQUIRE(arr[0] == 1 + 2);
    REQUIRE(arr[1] == 2 + 2);
    REQUIRE(arr[2] == 1 + 3);
    REQUIRE(arr[3] == 2 + 3);
    REQUIRE(arr[4] == 1 + 4);
    REQUIRE(arr[5] == 2 + 4);
  }

  SECTION("Test use of std::transform") {
    auto func = [](const std::array<int, 2> &indices) -> int {
      return indices[0] + indices[1];
    };
    auto index = arr.get_index();
    auto data = arr.get_data();
    std::transform(index.begin(), index.end(), data.begin(), func);
  }
}

TEST_CASE("Array::set_data - move semantics enabled", "[array][set_data][move]") {
  // Create a vector with known contents
  std::vector<double> original(100, 3.14);
  original[0] = 1.0;
  original[99] = 2.0;
  std::vector<double> moved_from = original;

  // Create array and move data in
  Array<double, 3> arr({10, 10, 1});
  arr.set_data(std::move(moved_from));

  // Verify data was moved correctly into array
  REQUIRE(arr.get_data().size() == 100);
  REQUIRE(arr.get_data()[0] == 1.0);
  REQUIRE(arr.get_data()[99] == 2.0);

  // Verify moved-from vector is now empty (actual move occurred)
  REQUIRE(moved_from.empty());

  // Verify still has moved-from state (not just size 0, but capacity 0)
  REQUIRE(moved_from.capacity() == 0);
}
