// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <array>
#include <iostream>
#include <sstream>

#include <catch2/catch_test_macros.hpp>

#include "openpfc/multi_index.hpp"

using namespace pfc;

TEST_CASE("MultiIndex - iteration over 3D grid", "[multi_index][unit]") {
  SECTION("Produces correct indices and linear indices") {
    std::stringstream output;
    std::array<int, 3> size = {2, 2, 2};
    std::array<int, 3> offset = {1, 1, 1};
    MultiIndex<3> index(size, offset);

    for (auto it = index.begin(); it != index.end(); ++it) {
      output << it << ", linear index = " << it.get_linear_index() << "\n";
    }

    // Define the expected output string
    std::string expectedOutput = R"EXPECTED_OUTPUT({1, 1, 1}, linear index = 0
{2, 1, 1}, linear index = 1
{1, 2, 1}, linear index = 2
{2, 2, 1}, linear index = 3
{1, 1, 2}, linear index = 4
{2, 1, 2}, linear index = 5
{1, 2, 2}, linear index = 6
{2, 2, 2}, linear index = 7
)EXPECTED_OUTPUT";

    // Compare the actual and expected outputs
    REQUIRE(output.str() == expectedOutput);
  }

  SECTION("Can be used to fill vector with computed values") {
    std::array<int, 8> data{};
    std::array<int, 3> size = {2, 2, 2};
    std::array<int, 3> offset = {1, 1, 1};
    MultiIndex index(size, offset);

    for (auto it = index.begin(); it != index.end(); ++it) {
      const auto [i, j, k] = *it;
      data[it.get_linear_index()] = 2 * i + j + 3 * k;
    }

    std::array<int, 8> expectedData = {6, 8, 7, 9, 9, 11, 10, 12};
    REQUIRE(data == expectedData);
  }
}

TEST_CASE("MultiIndex - 2D grid operations", "[multi_index][unit]") {
  SECTION("Can fill 2D array with selective region") {
    std::array<int, 25> arr{};
    std::array<int, 2> size = {5, 5};
    std::array<int, 2> offset = {3, 3};
    MultiIndex<2> index2d(size, offset);
    std::array<int, 2> start_index{4, 4};
    std::array<int, 2> end_index{6, 6};

    for (int j = start_index[1]; j <= end_index[1]; ++j) {
      for (int i = start_index[0]; i <= end_index[0]; ++i) {
        arr[index2d.to_linear({i, j})] = 1;
      }
    }

    std::string expectedOutput = R"EXPECTED_OUTPUT(0 0 0 0 0 
0 1 1 1 0 
0 1 1 1 0 
0 1 1 1 0 
0 0 0 0 0 
)EXPECTED_OUTPUT";

    std::stringstream output;
    for (size_t i = 0; i < arr.size(); ++i) {
      output << arr[i] << ' ';
      if ((i + 1) % 5 == 0) {
        output << '\n';
      }
    }

    REQUIRE(output.str() == expectedOutput);
  }
}
