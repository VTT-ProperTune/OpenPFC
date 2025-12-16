// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_sparse_vector_neighbor_exchange.cpp
 * @brief Comprehensive tests for SparseVector neighbor exchange via MPI
 *
 * Tests cover:
 * - Bidirectional exchange between neighbors
 * - Ring topology (each rank exchanges with left and right neighbors)
 * - Multiple simultaneous exchanges
 * - Large data exchange
 * - Empty sparse vector exchange
 * - Different exchange patterns
 */

#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <numeric>
#include <openpfc/core/exchange.hpp>
#include <openpfc/core/sparse_vector.hpp>
#include <openpfc/core/sparse_vector_ops.hpp>
#include <vector>

using namespace pfc;
using Catch::Approx;

// Helper to get rank and size
// Note: MPI is already initialized by runtests.cpp
std::pair<int, int> get_mpi_info() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return {rank, size};
}

TEST_CASE("Neighbor exchange - Bidirectional 2 processes",
          "[SparseVector][MPI][neighbor][bidirectional]") {

  auto [rank, size] = get_mpi_info();

  if (size < 2) {

    SKIP("This test requires at least 2 MPI processes");
  }

  if (rank == 0) {
    // Rank 0 sends to rank 1
    std::vector<size_t> indices = {0, 2, 4};
    std::vector<double> data = {10.0, 20.0, 30.0};
    auto sparse_send = sparsevector::create<double>(indices, data);
    exchange::send(sparse_send, 0, 1, MPI_COMM_WORLD);

    // Rank 0 receives from rank 1
    auto sparse_recv = sparsevector::create<double>(0);
    exchange::receive(sparse_recv, 1, 0, MPI_COMM_WORLD);

    REQUIRE(sparsevector::get_size(sparse_recv) == 3);
    auto recv_data = sparsevector::get_data(sparse_recv);
    REQUIRE(recv_data[0] == Approx(100.0).margin(1e-10));
    REQUIRE(recv_data[1] == Approx(200.0).margin(1e-10));
    REQUIRE(recv_data[2] == Approx(300.0).margin(1e-10));
  } else if (rank == 1) {
    // Rank 1 receives from rank 0
    auto sparse_recv = sparsevector::create<double>(0);
    exchange::receive(sparse_recv, 0, 1, MPI_COMM_WORLD);

    REQUIRE(sparsevector::get_size(sparse_recv) == 3);
    auto recv_data = sparsevector::get_data(sparse_recv);
    REQUIRE(recv_data[0] == Approx(10.0).margin(1e-10));
    REQUIRE(recv_data[1] == Approx(20.0).margin(1e-10));
    REQUIRE(recv_data[2] == Approx(30.0).margin(1e-10));

    // Rank 1 sends to rank 0
    std::vector<size_t> indices = {1, 3, 5};
    std::vector<double> data = {100.0, 200.0, 300.0};
    auto sparse_send = sparsevector::create<double>(indices, data);
    exchange::send(sparse_send, 1, 0, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_CASE("Neighbor exchange - Ring topology",
          "[SparseVector][MPI][neighbor][ring]") {
  auto [rank, size] = get_mpi_info();

  if (size < 3) {
    SKIP("This test requires at least 3 MPI processes");
  }

  // Each rank sends to next (right neighbor) and receives from previous (left
  // neighbor)
  int right_neighbor = (rank + 1) % size;
  int left_neighbor = (rank - 1 + size) % size;

  // Create data to send
  std::vector<size_t> send_indices = {static_cast<size_t>(rank * 10),
                                      static_cast<size_t>(rank * 10 + 1),
                                      static_cast<size_t>(rank * 10 + 2)};
  std::vector<double> send_data = {static_cast<double>(rank * 100),
                                   static_cast<double>(rank * 100 + 10),
                                   static_cast<double>(rank * 100 + 20)};
  auto sparse_send = sparsevector::create<double>(send_indices, send_data);

  // Send to right neighbor
  exchange::send(sparse_send, rank, right_neighbor, MPI_COMM_WORLD);

  // Receive from left neighbor
  auto sparse_recv = sparsevector::create<double>(0);
  exchange::receive(sparse_recv, left_neighbor, rank, MPI_COMM_WORLD);

  // Verify received data
  REQUIRE(sparsevector::get_size(sparse_recv) == 3);
  auto recv_data = sparsevector::get_data(sparse_recv);
  double expected_base = static_cast<double>(left_neighbor * 100);
  REQUIRE(recv_data[0] == Approx(expected_base).margin(1e-10));
  REQUIRE(recv_data[1] == Approx(expected_base + 10.0).margin(1e-10));
  REQUIRE(recv_data[2] == Approx(expected_base + 20.0).margin(1e-10));

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_CASE("Neighbor exchange - Data-only runtime phase",
          "[SparseVector][MPI][neighbor][runtime]") {

  auto [rank, size] = get_mpi_info();

  if (size < 2) {

    SKIP("This test requires at least 2 MPI processes");
  }

  int partner = (rank == 0) ? 1 : 0;

  // Setup: Exchange indices once
  std::vector<size_t> indices = {0, 1, 2, 3, 4};
  auto sparse = sparsevector::create<double>(indices);

  if (rank == 0) {
    exchange::send(sparse, 0, 1, MPI_COMM_WORLD);
  } else {
    exchange::receive(sparse, 0, 1, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Runtime: Exchange data multiple times
  const int num_steps = 10;
  for (int step = 0; step < num_steps; ++step) {
    if (rank == 0) {
      // Update data
      std::vector<double> new_data = {
          static_cast<double>(step * 10), static_cast<double>(step * 10 + 1),
          static_cast<double>(step * 10 + 2), static_cast<double>(step * 10 + 3),
          static_cast<double>(step * 10 + 4)};
      sparsevector::set_data(sparse, new_data);
      exchange::send_data(sparse, 0, 1, MPI_COMM_WORLD);
    } else {
      exchange::receive_data(sparse, 0, 1, MPI_COMM_WORLD);
      auto data = sparsevector::get_data(sparse);
      REQUIRE(data[0] == Approx(step * 10.0).margin(1e-10));
      REQUIRE(data[1] == Approx(step * 10.0 + 1.0).margin(1e-10));
      REQUIRE(data[2] == Approx(step * 10.0 + 2.0).margin(1e-10));
      REQUIRE(data[3] == Approx(step * 10.0 + 3.0).margin(1e-10));
      REQUIRE(data[4] == Approx(step * 10.0 + 4.0).margin(1e-10));
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

TEST_CASE("Neighbor exchange - Large data", "[SparseVector][MPI][neighbor][large]") {

  auto [rank, size] = get_mpi_info();

  if (size < 2) {

    SKIP("This test requires at least 2 MPI processes");
  }

  const size_t large_size = 10000;
  std::vector<size_t> indices(large_size);
  std::vector<double> data(large_size);

  if (rank == 0) {
    // Create large sparse vector
    std::iota(indices.begin(), indices.end(), 0);
    std::iota(data.begin(), data.end(), 1.0);

    auto sparse = sparsevector::create<double>(indices, data);
    exchange::send(sparse, 0, 1, MPI_COMM_WORLD);
  } else if (rank == 1) {
    auto sparse = sparsevector::create<double>(0);
    exchange::receive(sparse, 0, 1, MPI_COMM_WORLD);

    REQUIRE(sparsevector::get_size(sparse) == large_size);
    auto recv_data = sparsevector::get_data(sparse);
    for (size_t i = 0; i < large_size; ++i) {
      REQUIRE(recv_data[i] == Approx(1.0 + i).margin(1e-10));
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_CASE("Neighbor exchange - Empty sparse vector",
          "[SparseVector][MPI][neighbor][empty]") {

  auto [rank, size] = get_mpi_info();

  if (size < 2) {

    SKIP("This test requires at least 2 MPI processes");
  }

  if (rank == 0) {
    // Send empty sparse vector
    auto sparse = sparsevector::create<double>(0);
    exchange::send(sparse, 0, 1, MPI_COMM_WORLD);
  } else if (rank == 1) {
    // Receive empty sparse vector
    auto sparse = sparsevector::create<double>(0);
    exchange::receive(sparse, 0, 1, MPI_COMM_WORLD);
    REQUIRE(sparsevector::get_size(sparse) == 0);
    REQUIRE(sparse.empty());
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_CASE("Neighbor exchange - Multiple neighbors simultaneously",
          "[SparseVector][MPI][neighbor][multiple]") {

  auto [rank, size] = get_mpi_info();

  if (size < 4) {

    SKIP("This test requires at least 4 MPI processes");
  }

  // Each rank exchanges with two neighbors
  int left_neighbor = (rank - 1 + size) % size;
  int right_neighbor = (rank + 1) % size;

  // Create data for each direction
  std::vector<size_t> left_indices = {static_cast<size_t>(rank * 100),
                                      static_cast<size_t>(rank * 100 + 1)};
  std::vector<double> left_data = {static_cast<double>(rank * 1000),
                                   static_cast<double>(rank * 1000 + 100)};
  auto sparse_left = sparsevector::create<double>(left_indices, left_data);

  std::vector<size_t> right_indices = {static_cast<size_t>(rank * 200),
                                       static_cast<size_t>(rank * 200 + 1)};
  std::vector<double> right_data = {static_cast<double>(rank * 2000),
                                    static_cast<double>(rank * 2000 + 200)};
  auto sparse_right = sparsevector::create<double>(right_indices, right_data);

  // Send to both neighbors
  exchange::send(sparse_left, rank, left_neighbor, MPI_COMM_WORLD, 100);
  exchange::send(sparse_right, rank, right_neighbor, MPI_COMM_WORLD, 200);

  // Receive from both neighbors
  auto recv_left = sparsevector::create<double>(0);
  auto recv_right = sparsevector::create<double>(0);
  exchange::receive(recv_left, left_neighbor, rank, MPI_COMM_WORLD, 100);
  exchange::receive(recv_right, right_neighbor, rank, MPI_COMM_WORLD, 200);

  // Verify received data
  REQUIRE(sparsevector::get_size(recv_left) == 2);
  REQUIRE(sparsevector::get_size(recv_right) == 2);

  auto left_data_recv = sparsevector::get_data(recv_left);
  auto right_data_recv = sparsevector::get_data(recv_right);

  double expected_left_base = static_cast<double>(left_neighbor * 1000);
  REQUIRE(left_data_recv[0] == Approx(expected_left_base).margin(1e-10));
  REQUIRE(left_data_recv[1] == Approx(expected_left_base + 100.0).margin(1e-10));

  double expected_right_base = static_cast<double>(right_neighbor * 2000);
  REQUIRE(right_data_recv[0] == Approx(expected_right_base).margin(1e-10));
  REQUIRE(right_data_recv[1] == Approx(expected_right_base + 200.0).margin(1e-10));

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_CASE("Neighbor exchange - 2D grid pattern (4 processes)",
          "[SparseVector][MPI][neighbor][grid]") {

  auto [rank, size] = get_mpi_info();

  if (size != 4) {

    SKIP("This test requires exactly 4 MPI processes");
  }

  // Arrange as 2x2 grid:
  // 0 -- 1
  // |    |
  // 2 -- 3
  // Each rank exchanges with horizontal and vertical neighbors

  std::vector<int> horizontal_neighbors = {1, 0, 3, 2}; // For ranks 0,1,2,3
  std::vector<int> vertical_neighbors = {2, 3, 0, 1};   // For ranks 0,1,2,3

  int h_neighbor = horizontal_neighbors[rank];
  int v_neighbor = vertical_neighbors[rank];

  // Create data for each direction
  std::vector<size_t> h_indices = {rank * 10};
  std::vector<double> h_data = {static_cast<double>(rank * 100)};
  auto sparse_h = sparsevector::create<double>(h_indices, h_data);

  std::vector<size_t> v_indices = {rank * 20};
  std::vector<double> v_data = {static_cast<double>(rank * 200)};
  auto sparse_v = sparsevector::create<double>(v_indices, v_data);

  // Send to neighbors
  exchange::send(sparse_h, rank, h_neighbor, MPI_COMM_WORLD, 10);
  exchange::send(sparse_v, rank, v_neighbor, MPI_COMM_WORLD, 20);

  // Receive from neighbors
  auto recv_h = sparsevector::create<double>(0);
  auto recv_v = sparsevector::create<double>(0);
  exchange::receive(recv_h, h_neighbor, rank, MPI_COMM_WORLD, 10);
  exchange::receive(recv_v, v_neighbor, rank, MPI_COMM_WORLD, 20);

  // Verify
  REQUIRE(sparsevector::get_size(recv_h) == 1);
  REQUIRE(sparsevector::get_size(recv_v) == 1);

  auto h_data_recv = sparsevector::get_data(recv_h);
  auto v_data_recv = sparsevector::get_data(recv_v);

  REQUIRE(h_data_recv[0] == Approx(h_neighbor * 100.0).margin(1e-10));
  REQUIRE(v_data_recv[0] == Approx(v_neighbor * 200.0).margin(1e-10));

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_CASE("Neighbor exchange - Data-only with different tags",
          "[SparseVector][MPI][neighbor][tags]") {

  auto [rank, size] = get_mpi_info();

  if (size < 2) {

    SKIP("This test requires at least 2 MPI processes");
  }

  int partner = (rank == 0) ? 1 : 0;

  // Setup with different tags
  std::vector<size_t> indices = {0, 1, 2};
  auto sparse1 = sparsevector::create<double>(indices);
  auto sparse2 = sparsevector::create<double>(indices);

  if (rank == 0) {
    exchange::send(sparse1, 0, 1, MPI_COMM_WORLD, 100);
    exchange::send(sparse2, 0, 1, MPI_COMM_WORLD, 200);
  } else {
    exchange::receive(sparse1, 0, 1, MPI_COMM_WORLD, 100);
    exchange::receive(sparse2, 0, 1, MPI_COMM_WORLD, 200);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Runtime: Exchange data with different tags simultaneously
  if (rank == 0) {
    sparsevector::set_data(sparse1, {1.0, 2.0, 3.0});
    sparsevector::set_data(sparse2, {10.0, 20.0, 30.0});
    exchange::send_data(sparse1, 0, 1, MPI_COMM_WORLD, 100);
    exchange::send_data(sparse2, 0, 1, MPI_COMM_WORLD, 200);
  } else {
    exchange::receive_data(sparse1, 0, 1, MPI_COMM_WORLD, 100);
    exchange::receive_data(sparse2, 0, 1, MPI_COMM_WORLD, 200);

    auto data1 = sparsevector::get_data(sparse1);
    auto data2 = sparsevector::get_data(sparse2);

    REQUIRE(data1[0] == Approx(1.0).margin(1e-10));
    REQUIRE(data1[1] == Approx(2.0).margin(1e-10));
    REQUIRE(data1[2] == Approx(3.0).margin(1e-10));

    REQUIRE(data2[0] == Approx(10.0).margin(1e-10));
    REQUIRE(data2[1] == Approx(20.0).margin(1e-10));
    REQUIRE(data2[2] == Approx(30.0).margin(1e-10));
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_CASE("Neighbor exchange - Gather-scatter round-trip via exchange",
          "[SparseVector][MPI][neighbor][roundtrip]") {

  auto [rank, size] = get_mpi_info();

  if (size < 2) {

    SKIP("This test requires at least 2 MPI processes");
  }

  int partner = (rank == 0) ? 1 : 0;

  // Rank 0: Create source array, gather into sparse, send
  // Rank 1: Receive, scatter to destination, verify

  if (rank == 0) {
    std::vector<double> source = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    std::vector<size_t> indices = {1, 3, 5, 7};
    auto sparse = sparsevector::create<double>(indices);
    gather(sparse, source);
    exchange::send(sparse, 0, 1, MPI_COMM_WORLD);
  } else {
    auto sparse = sparsevector::create<double>(0);
    exchange::receive(sparse, 0, 1, MPI_COMM_WORLD);

    std::vector<double> dest(8, 0.0);
    scatter(sparse, dest);

    REQUIRE(dest[1] == 2.0);
    REQUIRE(dest[3] == 4.0);
    REQUIRE(dest[5] == 6.0);
    REQUIRE(dest[7] == 8.0);
    REQUIRE(dest[0] == 0.0);
    REQUIRE(dest[2] == 0.0);
    REQUIRE(dest[4] == 0.0);
    REQUIRE(dest[6] == 0.0);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}
