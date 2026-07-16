// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cassert>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/runtime/hip/sparse_vector_ops.hpp>

using namespace pfc;

TEST_CASE("Construct empty SparseVector", "[SparseVector (HIP)]") {
  auto vector = sparsevector::create<double, HipTag>(3);
  REQUIRE(get_size(vector) == 3);
  REQUIRE(!on_host(vector));
}

TEST_CASE("Update SparseVector", "[SparseVector (HIP)]") {
  auto vector = sparsevector::create<double, HipTag>(3);
  set_index(vector, {2, 3, 4});
  set_data(vector, {1.0, 2.0, 3.0});
  auto index = get_index(vector);
  auto data = get_data(vector);
  REQUIRE(index[0] == 2);
  REQUIRE(data[0] == 1.0);
}

TEST_CASE("Construct filled SparseVector", "[SparseVector (HIP)]") {
  std::vector<size_t> h_index = {2, 4, 6};
  std::vector<double> h_data = {1.0, 2.0, 3.0};
  auto vector = sparsevector::create<double, HipTag>(h_index, h_data);
  REQUIRE(get_size(vector) == 3);
}

TEST_CASE("Gather data from source", "[SparseVector (HIP)]") {
  std::vector<size_t> h_index = {0, 1, 3};
  auto vector = sparsevector::create<double, HipTag>(h_index);
  std::vector<double> h_big_data = {1.0, 2.0, 3.0, 4.0};
  
  // Copy host data to device
  double *d_big_data = nullptr;
  hipMalloc(&d_big_data, h_big_data.size() * sizeof(double));
  hipMemcpy(d_big_data, h_big_data.data(), h_big_data.size() * sizeof(double), 
            hipMemcpyHostToDevice);
  
  // Gather from device data
  gather(vector, d_big_data, h_big_data.size());
  
  // Verify gathered data
  std::vector<double> h_result(3);
  hipMemcpy(h_result.data(), get_data(vector).data(), 3 * sizeof(double), 
            hipMemcpyDeviceToHost);
  
  REQUIRE(h_result[0] == 1.0);
  REQUIRE(h_result[1] == 2.0);
  REQUIRE(h_result[2] == 4.0);
  
  hipFree(d_big_data);
}

TEST_CASE("Scatter data to destination", "[SparseVector (HIP)]") {
  std::vector<size_t> h_indices = {0, 1, 3};
  std::vector<double> h_data = {1.0, 2.0, 4.0};
  auto vector = sparsevector::create<double, HipTag>(h_indices, h_data);
  
  std::vector<double> h_big_data(4, 0.0);
  double *d_big_data = nullptr;
  hipMalloc(&d_big_data, h_big_data.size() * sizeof(double));
  hipMemcpy(d_big_data, h_big_data.data(), h_big_data.size() * sizeof(double), 
            hipMemcpyHostToDevice);
  
  // Scatter to device data
  scatter(vector, d_big_data, h_big_data.size());
  
  // Verify scattered data
  hipMemcpy(h_big_data.data(), d_big_data, h_big_data.size() * sizeof(double), 
            hipMemcpyDeviceToHost);
  
  REQUIRE(h_big_data[0] == 1.0);
  REQUIRE(h_big_data[1] == 2.0);
  REQUIRE(h_big_data[2] == 0.0);
  REQUIRE(h_big_data[3] == 4.0);
  
  hipFree(d_big_data);
}
