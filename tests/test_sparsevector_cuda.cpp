// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cassert>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thrust/device_vector.h>
#include <vector>

using namespace pfc;

TEST_CASE("Construct empty SparseVector", "[SparseVector (CUDA)]") {
  auto vector = sparsevector::create<double, CUDATag>(3);
  REQUIRE(get_size(vector) == 3);
  REQUIRE(!on_host(vector));
}

TEST_CASE("Update SparseVector", "[SparseVector (CUDA)]") {
  auto vector = sparsevector::create<double, CUDATag>(3);
  set_index(vector, {2, 3, 4});
  set_data(vector, {1.0, 2.0, 3.0});
  auto index = get_index(vector);
  auto data = get_data(vector);
  REQUIRE(index[0] == 2);
  REQUIRE(data[0] == 1.0);
}

TEST_CASE("Construct filled SparseVector", "[SparseVector (CUDA)]") {
  thrust::device_vector<size_t> index = {2, 4, 6};
  thrust::device_vector<double> data = {1.0, 2.0, 3.0};
  auto vector = sparsevector::create(index, data);
  REQUIRE(get_size(vector) == 3);
}

TEST_CASE("Gather data from source", "[SparseVector (CUDA)]") {
  thrust::device_vector<size_t> index = {0, 1, 3};
  auto vector = sparsevector::create<double>(index);
  thrust::device_vector<double> big_data = {1.0, 2.0, 3.0, 4.0};
  gather(vector, big_data);
  REQUIRE(get_data(vector) == {1.0, 2.0, 4.0});
}

TEST_CASE("Scatter data to destination", "[SparseVector (CUDA)]") {
  thrust::device_vector<size_t> indices = {0, 1, 3};
  auto vector = sparsevector::create<double, CUDATag>({0, 1, 3}, {1.0, 2.0, 4.0});
  thrust::device_vector<double> big_data = {0.0, 0.0, 0.0, 0.0};
  scatter(vector, big_data);
  REQUIRE(big_data == thrust::device_vector<double>({1.0, 2.0, 0.0, 4.0}));
}
