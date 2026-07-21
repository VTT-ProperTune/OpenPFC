// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <vector>

#include <openpfc/runtime/hip/sparse_vector_hip.hpp>
#include <openpfc/runtime/hip/sparse_vector_ops.hpp>

using namespace pfc;
using backend::HipTag;

TEST_CASE("Construct empty SparseVector", "[SparseVector (HIP)]") {
  auto vector = sparsevector::create<double, HipTag>(3);
  REQUIRE(sparsevector::get_size(vector) == 3);
  REQUIRE(!core::on_host(vector));
}

TEST_CASE("Construct filled SparseVector", "[SparseVector (HIP)]") {
  std::vector<size_t> h_index = {2, 4, 6};
  std::vector<double> h_data = {1.0, 2.0, 3.0};
  auto vector = sparsevector::create<double, HipTag>(h_index, h_data);
  REQUIRE(sparsevector::get_size(vector) == 3);
}

TEST_CASE("Gather data from source", "[SparseVector (HIP)]") {
  std::vector<size_t> h_index = {0, 1, 3};
  auto vector = sparsevector::create<double, HipTag>(h_index);
  std::vector<double> h_big_data = {1.0, 2.0, 3.0, 4.0};

  double *d_big_data = nullptr;
  REQUIRE(hipMalloc(&d_big_data, h_big_data.size() * sizeof(double)) ==
          hipSuccess);
  REQUIRE(hipMemcpy(d_big_data, h_big_data.data(),
                    h_big_data.size() * sizeof(double),
                    hipMemcpyHostToDevice) == hipSuccess);

  gather(vector, d_big_data, h_big_data.size());

  std::vector<double> h_result(3);
  REQUIRE(hipMemcpy(h_result.data(), vector.data().data(), 3 * sizeof(double),
                    hipMemcpyDeviceToHost) == hipSuccess);

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
  REQUIRE(hipMalloc(&d_big_data, h_big_data.size() * sizeof(double)) ==
          hipSuccess);
  REQUIRE(hipMemcpy(d_big_data, h_big_data.data(),
                    h_big_data.size() * sizeof(double),
                    hipMemcpyHostToDevice) == hipSuccess);

  scatter(vector, d_big_data, h_big_data.size());

  REQUIRE(hipMemcpy(h_big_data.data(), d_big_data,
                    h_big_data.size() * sizeof(double),
                    hipMemcpyDeviceToHost) == hipSuccess);

  REQUIRE(h_big_data[0] == 1.0);
  REQUIRE(h_big_data[1] == 2.0);
  REQUIRE(h_big_data[2] == 0.0);
  REQUIRE(h_big_data[3] == 4.0);

  hipFree(d_big_data);
}

TEST_CASE("Gather OOB throws", "[SparseVector (HIP)][error]") {
  std::vector<size_t> h_index = {0, 4}; // 4 >= dense length 4
  auto vector = sparsevector::create<double, HipTag>(h_index);
  std::vector<double> h_big_data = {1.0, 2.0, 3.0, 4.0};

  double *d_big_data = nullptr;
  REQUIRE(hipMalloc(&d_big_data, h_big_data.size() * sizeof(double)) ==
          hipSuccess);
  REQUIRE(hipMemcpy(d_big_data, h_big_data.data(),
                    h_big_data.size() * sizeof(double),
                    hipMemcpyHostToDevice) == hipSuccess);

  REQUIRE_THROWS_AS(gather(vector, d_big_data, h_big_data.size()),
                    std::runtime_error);

  hipFree(d_big_data);
}

TEST_CASE("Scatter OOB throws", "[SparseVector (HIP)][error]") {
  std::vector<size_t> h_indices = {0, 4}; // 4 >= dense length 4
  std::vector<double> h_data = {1.0, 2.0};
  auto vector = sparsevector::create<double, HipTag>(h_indices, h_data);

  std::vector<double> h_big_data(4, 0.0);
  double *d_big_data = nullptr;
  REQUIRE(hipMalloc(&d_big_data, h_big_data.size() * sizeof(double)) ==
          hipSuccess);
  REQUIRE(hipMemcpy(d_big_data, h_big_data.data(),
                    h_big_data.size() * sizeof(double),
                    hipMemcpyHostToDevice) == hipSuccess);

  REQUIRE_THROWS_AS(scatter(vector, d_big_data, h_big_data.size()),
                    std::runtime_error);

  hipFree(d_big_data);
}

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }
