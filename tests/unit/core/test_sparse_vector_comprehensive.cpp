// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_sparse_vector_comprehensive.cpp
 * @brief Comprehensive unit tests for SparseVector core functionality
 *
 * Tests cover:
 * - Construction and basic operations
 * - Gather/scatter with various scenarios
 * - Edge cases and error conditions
 * - Different data types
 * - Large data sets
 */

#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <numeric>
#include <openpfc/core/sparse_vector.hpp>
#include <openpfc/core/sparse_vector_ops.hpp>
#include <stdexcept>
#include <vector>

using namespace pfc;
using Catch::Approx;

TEST_CASE("SparseVector - Large index set", "[SparseVector][core][large]") {
  const size_t N = 10000;
  std::vector<size_t> indices(N);
  std::iota(indices.begin(), indices.end(), 0);
  std::reverse(indices.begin(), indices.end()); // Unsorted

  auto sparse = core::SparseVector<backend::CpuTag, double>(indices);

  REQUIRE(sparse.size() == N);
  REQUIRE(sparse.is_sorted());

  auto retrieved = sparsevector::get_index(sparse);
  // Spot check: verify first, middle, and last elements are sorted
  REQUIRE(retrieved[0] == 0);
  REQUIRE(retrieved[N / 2] == N / 2);
  REQUIRE(retrieved[N - 1] == N - 1);
  // Verify size matches
  REQUIRE(retrieved.size() == N);
}

TEST_CASE("SparseVector - Gather from large array",
          "[SparseVector][gather][large]") {
  const size_t array_size = 50000;
  const size_t sparse_size = 1000;

  std::vector<double> source(array_size);
  std::iota(source.begin(), source.end(), 0.0);

  // Select every 50th element
  std::vector<size_t> indices;
  for (size_t i = 0; i < sparse_size; ++i) {
    indices.push_back(i * 50);
  }

  auto sparse = sparsevector::create<double>(indices);
  gather(sparse, source);

  auto data = sparsevector::get_data(sparse);
  REQUIRE(data.size() == sparse_size);
  // Spot check: verify first, middle, and last elements
  REQUIRE(data[0] == Approx(0.0).margin(1e-10));
  REQUIRE(data[sparse_size / 2] == Approx((sparse_size / 2) * 50.0).margin(1e-10));
  REQUIRE(data[sparse_size - 1] == Approx((sparse_size - 1) * 50.0).margin(1e-10));
}

TEST_CASE("SparseVector - Scatter to large array",
          "[SparseVector][scatter][large]") {
  const size_t array_size = 100000;
  const size_t sparse_size = 5000;

  // Create sparse vector with values
  std::vector<size_t> indices;
  std::vector<double> values;
  for (size_t i = 0; i < sparse_size; ++i) {
    indices.push_back(i * 20);
    values.push_back(100.0 + i);
  }

  auto sparse = sparsevector::create<double>(indices, values);
  std::vector<double> dest(array_size, 0.0);

  scatter(sparse, dest);

  // Spot check scattered values
  REQUIRE(dest[0] == Approx(100.0).margin(1e-10));
  REQUIRE(dest[20] == Approx(101.0).margin(1e-10));
  REQUIRE(dest[(sparse_size - 1) * 20] ==
          Approx(100.0 + sparse_size - 1).margin(1e-10));

  // Spot check untouched positions
  REQUIRE(dest[1] == 0.0);
  REQUIRE(dest[19] == 0.0);
  REQUIRE(dest[21] == 0.0);
}

TEST_CASE("SparseVector - Gather with out-of-bounds check",
          "[SparseVector][gather][error]") {
  std::vector<size_t> indices = {0, 5, 10};
  auto sparse = sparsevector::create<double>(indices);

  std::vector<double> source = {1.0, 2.0, 3.0, 4.0,
                                5.0}; // Size 5, but index 10 is out of bounds

  REQUIRE_THROWS_AS(gather(sparse, source), std::runtime_error);
}

TEST_CASE("SparseVector - Scatter with out-of-bounds check",
          "[SparseVector][scatter][error]") {
  std::vector<size_t> indices = {0, 5, 10};
  std::vector<double> values = {1.0, 2.0, 3.0};
  auto sparse = sparsevector::create<double>(indices, values);

  std::vector<double> dest(5, 0.0); // Size 5, but index 10 is out of bounds

  REQUIRE_THROWS_AS(scatter(sparse, dest), std::runtime_error);
}

TEST_CASE("SparseVector - Empty gather and scatter",
          "[SparseVector][gather][scatter][edge]") {
  auto sparse = sparsevector::create<double>(0); // Empty
  std::vector<double> source = {1.0, 2.0, 3.0};
  std::vector<double> dest(3, 0.0);

  // Should not throw or crash
  REQUIRE_NOTHROW(gather(sparse, source));
  REQUIRE_NOTHROW(scatter(sparse, dest));

  // Destination should be unchanged
  REQUIRE(dest[0] == 0.0);
  REQUIRE(dest[1] == 0.0);
  REQUIRE(dest[2] == 0.0);
}

TEST_CASE("SparseVector - Single element", "[SparseVector][core][edge]") {
  auto sparse = sparsevector::create<double>({42}, {3.14});
  REQUIRE(sparse.size() == 1);
  REQUIRE(sparse.is_sorted());

  auto idx = sparsevector::get_index(sparse);
  auto data = sparsevector::get_data(sparse);
  REQUIRE(idx[0] == 42);
  REQUIRE(data[0] == Approx(3.14).margin(1e-10));
}

TEST_CASE("SparseVector - Gather scatter with single element",
          "[SparseVector][gather][scatter][edge]") {
  auto sparse = sparsevector::create<double>({5});

  std::vector<double> source = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  gather(sparse, source);

  auto data = sparsevector::get_data(sparse);
  REQUIRE(data[0] == 5.0);

  std::vector<double> dest(7, 0.0);
  scatter(sparse, dest);
  REQUIRE(dest[5] == 5.0);
  REQUIRE(dest[0] == 0.0);
  REQUIRE(dest[6] == 0.0);
}

TEST_CASE("SparseVector - Float type", "[SparseVector][core][types]") {
  std::vector<size_t> indices = {1, 3, 5};
  std::vector<float> values = {1.5f, 3.5f, 5.5f};
  auto sparse = core::SparseVector<backend::CpuTag, float>(indices, values);

  REQUIRE(sparse.size() == 3);
  auto data = sparsevector::get_data(sparse);
  REQUIRE(data[0] == Approx(1.5f).margin(1e-5f));
  REQUIRE(data[1] == Approx(3.5f).margin(1e-5f));
  REQUIRE(data[2] == Approx(5.5f).margin(1e-5f));
}

TEST_CASE("SparseVector - Gather scatter with float",
          "[SparseVector][gather][scatter][types]") {
  auto sparse = sparsevector::create<float>({0, 2, 4});

  std::vector<float> source = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
  gather(sparse, source);

  auto data = sparsevector::get_data(sparse);
  REQUIRE(data[0] == Approx(1.1f).margin(1e-5f));
  REQUIRE(data[1] == Approx(3.3f).margin(1e-5f));
  REQUIRE(data[2] == Approx(5.5f).margin(1e-5f));

  std::vector<float> dest(5, 0.0f);
  scatter(sparse, dest);
  REQUIRE(dest[0] == Approx(1.1f).margin(1e-5f));
  REQUIRE(dest[2] == Approx(3.3f).margin(1e-5f));
  REQUIRE(dest[4] == Approx(5.5f).margin(1e-5f));
}

TEST_CASE("SparseVector - Gather from initializer list",
          "[SparseVector][gather][convenience]") {
  auto sparse = sparsevector::create<double>({1, 3, 5});

  // Using initializer list convenience overload
  gather(sparse, {10.0, 20.0, 30.0, 40.0, 50.0, 60.0});

  auto data = sparsevector::get_data(sparse);
  REQUIRE(data[0] == 20.0); // index 1
  REQUIRE(data[1] == 40.0); // index 3
  REQUIRE(data[2] == 60.0); // index 5
}

TEST_CASE("SparseVector - Multiple gather operations",
          "[SparseVector][gather][multiple]") {
  auto sparse = sparsevector::create<double>({0, 2, 4});

  std::vector<double> source1 = {1.0, 2.0, 3.0, 4.0, 5.0};
  gather(sparse, source1);
  auto data1 = sparsevector::get_data(sparse);
  REQUIRE(data1[0] == 1.0);
  REQUIRE(data1[1] == 3.0);
  REQUIRE(data1[2] == 5.0);

  std::vector<double> source2 = {10.0, 20.0, 30.0, 40.0, 50.0};
  gather(sparse, source2);
  auto data2 = sparsevector::get_data(sparse);
  REQUIRE(data2[0] == 10.0);
  REQUIRE(data2[1] == 30.0);
  REQUIRE(data2[2] == 50.0);
}

TEST_CASE("SparseVector - Scatter accumulation",
          "[SparseVector][scatter][accumulation]") {
  auto sparse1 = sparsevector::create<double>({1, 3}, {10.0, 30.0});
  auto sparse2 = sparsevector::create<double>({2, 3}, {20.0, 40.0});

  std::vector<double> dest(5, 0.0);

  scatter(sparse1, dest);
  REQUIRE(dest[1] == 10.0);
  REQUIRE(dest[3] == 30.0);

  scatter(sparse2, dest);
  REQUIRE(dest[1] == 10.0); // Unchanged
  REQUIRE(dest[2] == 20.0);
  REQUIRE(dest[3] == 40.0); // Overwritten
}

TEST_CASE("SparseVector - Gather scatter round-trip consistency",
          "[SparseVector][gather][scatter][consistency]") {
  const size_t N = 1000;
  std::vector<size_t> indices;
  for (size_t i = 0; i < N; i += 3) {
    indices.push_back(i);
  }

  auto sparse = sparsevector::create<double>(indices);

  // Create source with known pattern
  std::vector<double> source(N);
  for (size_t i = 0; i < N; ++i) {
    source[i] = static_cast<double>(i * i);
  }

  gather(sparse, source);

  // Scatter to destination
  std::vector<double> dest(N, -1.0);
  scatter(sparse, dest);

  // Spot check consistency: verify a few scattered and untouched positions
  REQUIRE(dest[0] == Approx(0.0).margin(1e-10));  // i=0, i%3==0
  REQUIRE(dest[3] == Approx(9.0).margin(1e-10));  // i=3, i%3==0
  REQUIRE(dest[6] == Approx(36.0).margin(1e-10)); // i=6, i%3==0
  REQUIRE(dest[1] == -1.0);                       // i=1, i%3!=0
  REQUIRE(dest[2] == -1.0);                       // i=2, i%3!=0
  REQUIRE(dest[4] == -1.0);                       // i=4, i%3!=0
}

TEST_CASE("SparseVector - Index sorting preserves data correspondence",
          "[SparseVector][core][sorting]") {
  // Create with unsorted indices and corresponding data
  std::vector<size_t> indices = {5, 1, 3, 7, 2};
  std::vector<double> data = {50.0, 10.0, 30.0, 70.0, 20.0};

  auto sparse = core::SparseVector<backend::CpuTag, double>(indices, data);

  REQUIRE(sparse.is_sorted());

  auto sorted_indices = sparsevector::get_index(sparse);
  auto sorted_data = sparsevector::get_data(sparse);

  // Verify indices are sorted
  REQUIRE(sorted_indices[0] == 1);
  REQUIRE(sorted_indices[1] == 2);
  REQUIRE(sorted_indices[2] == 3);
  REQUIRE(sorted_indices[3] == 5);
  REQUIRE(sorted_indices[4] == 7);

  // Verify data corresponds to sorted indices
  REQUIRE(sorted_data[0] == 10.0); // index 1
  REQUIRE(sorted_data[1] == 20.0); // index 2
  REQUIRE(sorted_data[2] == 30.0); // index 3
  REQUIRE(sorted_data[3] == 50.0); // index 5
  REQUIRE(sorted_data[4] == 70.0); // index 7
}

TEST_CASE("SparseVector - set_index and set_data for testing",
          "[SparseVector][testing][api]") {
  auto sparse = sparsevector::create<double>(3);

  sparsevector::set_index(sparse, {5, 2, 8});
  auto indices = sparsevector::get_index(sparse);
  REQUIRE(indices[0] == 2); // Sorted
  REQUIRE(indices[1] == 5);
  REQUIRE(indices[2] == 8);

  sparsevector::set_data(sparse, {20.0, 50.0, 80.0});
  auto data = sparsevector::get_data(sparse);
  REQUIRE(data[0] == 20.0);
  REQUIRE(data[1] == 50.0);
  REQUIRE(data[2] == 80.0);
}

TEST_CASE("SparseVector - set_data size mismatch error",
          "[SparseVector][testing][error]") {
  auto sparse = sparsevector::create<double>(3);

  std::vector<double> wrong_size = {1.0, 2.0}; // Size 2, but sparse is size 3
  REQUIRE_THROWS_AS(sparsevector::set_data(sparse, wrong_size), std::runtime_error);
}

TEST_CASE("SparseVector - get_entry accessor", "[SparseVector][api]") {
  std::vector<size_t> indices = {2, 4, 6};
  std::vector<double> data = {20.0, 40.0, 60.0};
  auto sparse = sparsevector::create<double>(indices, data);

  auto entry0 = sparsevector::get_entry(sparse, 0);
  REQUIRE(entry0.first == 2);
  REQUIRE(entry0.second == 20.0);

  auto entry1 = sparsevector::get_entry(sparse, 1);
  REQUIRE(entry1.first == 4);
  REQUIRE(entry1.second == 40.0);

  auto entry2 = sparsevector::get_entry(sparse, 2);
  REQUIRE(entry2.first == 6);
  REQUIRE(entry2.second == 60.0);
}
