/*

OpenPFC, a simulation software for the phase field crystal method.
Copyright (C) 2024 VTT Technical Research Centre of Finland Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.

*/

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <openpfc/multi_index.hpp>
#include <sstream>

using namespace pfc;

// Include the MultiIndex implementation and any other necessary headers

TEST_CASE("MultiIndex Example Test") {
  SECTION("Iterating through MultiIndex") {
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

  SECTION("Filling vector using MultiIndex") {
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

  SECTION("Filling two-dimensional data") {
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
