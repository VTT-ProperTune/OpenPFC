// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_sparse_vector.cpp
 * @brief Unit tests for SparseVector core functionality
 */

#include <catch2/catch_test_macros.hpp>
#include <openpfc/core/sparse_vector.hpp>
#include <openpfc/core/sparse_vector_ops.hpp>
#include <vector>

using namespace pfc;

TEST_CASE("SparseVector construction with indices", "[SparseVector][core]") {
  std::vector<size_t> indices = {5, 2, 8, 1};
  auto sparse = core::SparseVector<backend::CpuTag, double>(indices);

  REQUIRE(sparse.size() == 4);
  REQUIRE(sparse.is_sorted());

  // Indices should be sorted
  auto retrieved_indices = sparsevector::get_index(sparse);
  REQUIRE(retrieved_indices[0] == 1);
  REQUIRE(retrieved_indices[1] == 2);
  REQUIRE(retrieved_indices[2] == 5);
  REQUIRE(retrieved_indices[3] == 8);
}

TEST_CASE("SparseVector construction with indices and data",
          "[SparseVector][core]") {
  std::vector<size_t> indices = {3, 1, 4};
  std::vector<double> data = {30.0, 10.0, 40.0};
  auto sparse = core::SparseVector<backend::CpuTag, double>(indices, data);

  REQUIRE(sparse.size() == 3);
  REQUIRE(sparse.is_sorted());

  // Data should be reordered to match sorted indices
  auto retrieved_indices = sparsevector::get_index(sparse);
  auto retrieved_data = sparsevector::get_data(sparse);

  REQUIRE(retrieved_indices[0] == 1);
  REQUIRE(retrieved_data[0] == 10.0);

  REQUIRE(retrieved_indices[1] == 3);
  REQUIRE(retrieved_data[1] == 30.0);

  REQUIRE(retrieved_indices[2] == 4);
  REQUIRE(retrieved_data[2] == 40.0);
}

TEST_CASE("SparseVector gather operation", "[SparseVector][gather]") {
  std::vector<size_t> indices = {0, 2, 4};
  auto sparse = core::SparseVector<backend::CpuTag, double>(indices);

  std::vector<double> source = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  gather(sparse, source);

  auto data = sparsevector::get_data(sparse);
  REQUIRE(data[0] == 1.0); // index 0
  REQUIRE(data[1] == 3.0); // index 2
  REQUIRE(data[2] == 5.0); // index 4
}

TEST_CASE("SparseVector scatter operation", "[SparseVector][scatter]") {
  std::vector<size_t> indices = {1, 3, 5};
  std::vector<double> data = {10.0, 30.0, 50.0};
  auto sparse = core::SparseVector<backend::CpuTag, double>(indices, data);

  std::vector<double> dest(7, 0.0);
  scatter(sparse, dest);

  REQUIRE(dest[0] == 0.0);
  REQUIRE(dest[1] == 10.0);
  REQUIRE(dest[2] == 0.0);
  REQUIRE(dest[3] == 30.0);
  REQUIRE(dest[4] == 0.0);
  REQUIRE(dest[5] == 50.0);
  REQUIRE(dest[6] == 0.0);
}

TEST_CASE("SparseVector gather and scatter round-trip",
          "[SparseVector][gather][scatter]") {
  std::vector<size_t> indices = {0, 2, 4, 6};
  auto sparse = core::SparseVector<backend::CpuTag, double>(indices);

  // Gather from source
  std::vector<double> source = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  gather(sparse, source);

  // Scatter to destination
  std::vector<double> dest(8, 0.0);
  scatter(sparse, dest);

  // Verify values at sparse indices
  REQUIRE(dest[0] == 1.0);
  REQUIRE(dest[2] == 3.0);
  REQUIRE(dest[4] == 5.0);
  REQUIRE(dest[6] == 7.0);

  // Verify other positions are unchanged
  REQUIRE(dest[1] == 0.0);
  REQUIRE(dest[3] == 0.0);
  REQUIRE(dest[5] == 0.0);
  REQUIRE(dest[7] == 0.0);
}

TEST_CASE("SparseVector empty construction", "[SparseVector][core]") {
  auto sparse = core::SparseVector<backend::CpuTag, double>(0);
  REQUIRE(sparse.size() == 0);
  REQUIRE(sparse.empty());
}

TEST_CASE("SparseVector on_host check", "[SparseVector][core]") {
  auto sparse_cpu = core::SparseVector<backend::CpuTag, double>({1, 2, 3});
  REQUIRE(core::on_host(sparse_cpu) == true);

#if defined(OpenPFC_ENABLE_CUDA)
  auto sparse_cuda = core::SparseVector<backend::CudaTag, double>({1, 2, 3});
  REQUIRE(core::on_host(sparse_cuda) == false);
#endif
}

TEST_CASE("SparseVector with duplicate indices (should handle gracefully)",
          "[SparseVector][core]") {
  std::vector<size_t> indices = {2, 2, 4, 4, 4};
  auto sparse = core::SparseVector<backend::CpuTag, double>(indices);

  REQUIRE(sparse.size() == 5);
  REQUIRE(sparse.is_sorted());

  auto retrieved_indices = sparsevector::get_index(sparse);
  REQUIRE(retrieved_indices[0] == 2);
  REQUIRE(retrieved_indices[1] == 2);
  REQUIRE(retrieved_indices[2] == 4);
  REQUIRE(retrieved_indices[3] == 4);
  REQUIRE(retrieved_indices[4] == 4);
}
