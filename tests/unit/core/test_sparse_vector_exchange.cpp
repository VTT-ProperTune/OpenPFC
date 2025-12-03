// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_sparse_vector_exchange.cpp
 * @brief Unit tests for SparseVector MPI exchange operations
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <openpfc/core/exchange.hpp>
#include <openpfc/core/sparse_vector.hpp>
#include <vector>

using namespace pfc;
using Catch::Approx;

TEST_CASE("Exchange SparseVector indices and data",
          "[SparseVector][MPI][Exchange]") {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    SKIP("This test requires at least 2 MPI processes");
  }

  if (rank == 0) {
    // Rank 0: Create and send
    std::vector<size_t> indices = {0, 2, 4};
    std::vector<double> data = {1.0, 3.0, 5.0};
    auto sparse = sparsevector::create<double>(indices, data);

    exchange::send(sparse, 0, 1, MPI_COMM_WORLD);
  } else if (rank == 1) {
    // Rank 1: Receive
    auto sparse = sparsevector::create<double>(0); // Empty, will be resized
    exchange::receive(sparse, 0, 1, MPI_COMM_WORLD);

    REQUIRE(sparsevector::get_size(sparse) == 3);
    auto indices = sparsevector::get_index(sparse);
    auto data = sparsevector::get_data(sparse);

    // Indices should be sorted
    REQUIRE(indices[0] == 0);
    REQUIRE(indices[1] == 2);
    REQUIRE(indices[2] == 4);

    REQUIRE(data[0] == 1.0);
    REQUIRE(data[1] == 3.0);
    REQUIRE(data[2] == 5.0);
  }
}

TEST_CASE("Exchange only SparseVector data (indices known)",
          "[SparseVector][MPI][Exchange]") {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    SKIP("This test requires at least 2 MPI processes");
  }

  // Setup: Exchange indices once
  std::vector<size_t> indices = {1, 3, 5};
  auto sparse0 = sparsevector::create<double>(indices);
  auto sparse1 = sparsevector::create<double>(indices);

  if (rank == 0) {
    // Initial exchange of indices
    exchange::send(sparse0, 0, 1, MPI_COMM_WORLD);
  } else if (rank == 1) {
    exchange::receive(sparse1, 0, 1, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Runtime: Exchange only data (multiple times)
  for (int step = 0; step < 3; ++step) {
    if (rank == 0) {
      // Update data
      sparsevector::set_data(sparse0, {10.0 + step, 30.0 + step, 50.0 + step});
      exchange::send_data(sparse0, 0, 1, MPI_COMM_WORLD);
    } else if (rank == 1) {
      exchange::receive_data(sparse1, 0, 1, MPI_COMM_WORLD);
      auto data = sparsevector::get_data(sparse1);
      REQUIRE(data[0] == Approx(10.0 + step).margin(1e-10));
      REQUIRE(data[1] == Approx(30.0 + step).margin(1e-10));
      REQUIRE(data[2] == Approx(50.0 + step).margin(1e-10));
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

TEST_CASE("Exchange SparseVector with unsorted indices (should be sorted)",
          "[SparseVector][MPI][Exchange]") {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    SKIP("This test requires at least 2 MPI processes");
  }

  if (rank == 0) {
    // Send with unsorted indices - should be sorted automatically
    std::vector<size_t> indices = {5, 1, 3}; // Unsorted
    std::vector<double> data = {5.0, 1.0, 3.0};
    auto sparse = sparsevector::create<double>(indices, data);

    exchange::send(sparse, 0, 1, MPI_COMM_WORLD);
  } else if (rank == 1) {
    auto sparse = sparsevector::create<double>(0);
    exchange::receive(sparse, 0, 1, MPI_COMM_WORLD);

    REQUIRE(sparsevector::get_size(sparse) == 3);
    auto indices = sparsevector::get_index(sparse);
    auto data = sparsevector::get_data(sparse);

    // Should be sorted
    REQUIRE(indices[0] == 1);
    REQUIRE(indices[1] == 3);
    REQUIRE(indices[2] == 5);

    // Data should match sorted indices
    REQUIRE(data[0] == 1.0);
    REQUIRE(data[1] == 3.0);
    REQUIRE(data[2] == 5.0);
  }
}
