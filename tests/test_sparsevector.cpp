// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cassert>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pfc;

TEST_CASE("Construct empty SparseVector", "[SparseVector (Host)]") {
  auto vector = sparsevector::create<double, HostTag>(3);
  REQUIRE(get_size(vector) == 3);
  REQUIRE(on_host(vector));
}

TEST_CASE("Update SparseVector", "[SparseVector (Host)]") {
  auto vector = sparsevector::create<double>(3); // defaults to HostTag
  set_index(vector, {2, 3, 4});
  set_data(vector, {1.0, 2.0, 3.0});
  auto index = get_index(vector);
  auto data = get_data(vector);
  REQUIRE(index[0] == 2);
  REQUIRE(data[0] == 1.0);
}

TEST_CASE("Get single data from SparseVector", "[SparseVector (Host)]") {
  auto indices = std::vector<size_t>({2, 4, 6});
  auto data = std::vector<double>({1.0, 2.0, 3.0});
  auto vector = sparsevector::create(indices, data);
  // Check that the buffer has the expected properties
  REQUIRE(get_size(vector) == 3);
  REQUIRE(get_index(vector, 0) == 2);
  REQUIRE(get_data(vector, 0) == 1.0);
  REQUIRE(get_entry(vector, 0) == {2, 1.0});
}

TEST_CASE("Gather data from source", "[SparseVector (Host)]") {
  auto vector = sparsevector::create<double>({0, 1, 3});
  auto big_data = {1.0, 2.0, 3.0, 4.0};
  gather(vector, big_data);
  REQUIRE(get_entry(vector, 1) == {1, 2.0});
}

TEST_CASE("Scatter data to destination", "[SparseVector (Host)]") {
  auto vector = sparsevector::create<double>({0, 1, 3});
  auto big_data = {1.0, 2.0, 3.0, 4.0};
  scatter(vector, big_data);
  REQUIRE(get_entry(vector, 1) == {1, 2.0});
}
