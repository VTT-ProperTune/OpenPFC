// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>
#include <openpfc/array.hpp>

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
    for (int i = 0; i < 6; i++) REQUIRE(arr[i] == i + 1);
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
