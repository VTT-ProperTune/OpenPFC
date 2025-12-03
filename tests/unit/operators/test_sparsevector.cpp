// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <openpfc/core/exchange.hpp>
#include <openpfc/core/sparse_vector.hpp>
#include <openpfc/core/sparse_vector_ops.hpp>
#include <vector>

using namespace pfc;

TEST_CASE("Construct empty SparseVector", "[SparseVector (Host)]") {
  auto vector = sparsevector::create<double, sparsevector::HostTag>(3);
  REQUIRE(sparsevector::get_size(vector) == 3);
  REQUIRE(core::on_host(vector));
}

TEST_CASE("Update SparseVector", "[SparseVector (Host)]") {
  auto vector = sparsevector::create<double>(3); // defaults to CpuTag
  sparsevector::set_index(vector, {2, 3, 4});
  sparsevector::set_data(vector, {1.0, 2.0, 3.0});
  auto index = sparsevector::get_index(vector);
  auto data = sparsevector::get_data(vector);
  REQUIRE(index[0] == 2);
  REQUIRE(data[0] == 1.0);
}

TEST_CASE("Get single data from SparseVector", "[SparseVector (Host)]") {
  auto indices = std::vector<size_t>({2, 4, 6});
  auto data = std::vector<double>({1.0, 2.0, 3.0});
  auto vector = sparsevector::create(indices, data);
  // Check that the buffer has the expected properties
  REQUIRE(sparsevector::get_size(vector) == 3);
  REQUIRE(sparsevector::get_index(vector, 0) == 2);
  REQUIRE(sparsevector::get_data(vector, 0) == 1.0);
  auto entry = sparsevector::get_entry(vector, 0);
  REQUIRE(entry.first == 2);
  REQUIRE(entry.second == 1.0);
}

TEST_CASE("Gather data from source", "[SparseVector (Host)]") {
  auto vector = sparsevector::create<double>({0, 1, 3});
  std::vector<double> big_data = {1.0, 2.0, 3.0, 4.0};
  gather(vector, big_data);
  auto entry = sparsevector::get_entry(vector, 1);
  REQUIRE(entry.first == 1);
  REQUIRE(entry.second == 2.0);
}

TEST_CASE("Scatter data to destination", "[SparseVector (Host)]") {
  auto vector = sparsevector::create<double>({0, 1, 3}, {1.0, 2.0, 4.0});
  std::vector<double> big_data = {0.0, 0.0, 0.0, 0.0};
  scatter(vector, big_data);
  REQUIRE(big_data[0] == 1.0);
  REQUIRE(big_data[1] == 2.0);
  REQUIRE(big_data[2] == 0.0);
  REQUIRE(big_data[3] == 4.0);
}
