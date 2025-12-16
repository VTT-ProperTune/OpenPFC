// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_halo_pattern.cpp
 * @brief Unit tests for halo pattern creation from Decomposition
 */

#include <catch2/catch_test_macros.hpp>
#include <openpfc/core/decomposition.hpp>
#include <openpfc/core/halo_pattern.hpp>
#include <openpfc/core/sparse_vector_ops.hpp>
#include <openpfc/core/world.hpp>
#include <vector>

using namespace pfc;

TEST_CASE("Create send halo for +X direction", "[halo][pattern]") {
  auto world = world::create(GridSize({64), PhysicalOrigin(64), GridSpacing(64}));
  auto decomp = decomposition::create(world, {2, 2, 1}); // 2×2×1 = 4 ranks

  int rank = 0;
  int halo_width = 1;
  Int3 direction = {1, 0, 0}; // +X direction

  auto send_halo =
      halo::create_send_halo<backend::CpuTag>(decomp, rank, direction, halo_width);

  REQUIRE(send_halo.size() > 0);
  REQUIRE(send_halo.is_sorted());

  // Verify indices are valid local indices (spot check)
  auto indices = sparsevector::get_index(send_halo);
  auto local_world = decomposition::get_subworld(decomp, rank);
  auto local_size = world::get_size(local_world);
  size_t local_total = static_cast<size_t>(local_size[0]) *
                       static_cast<size_t>(local_size[1]) *
                       static_cast<size_t>(local_size[2]);

  // Spot check: verify first, middle, and last indices are valid
  if (!indices.empty()) {
    REQUIRE(indices[0] < local_total);
    REQUIRE(indices[indices.size() / 2] < local_total);
    REQUIRE(indices[indices.size() - 1] < local_total);
  }
}

TEST_CASE("Create recv halo for +X direction", "[halo][pattern]") {
  auto world = world::create(GridSize({64), PhysicalOrigin(64), GridSpacing(64}));
  auto decomp = decomposition::create(world, {2, 2, 1});

  int rank = 0;
  int halo_width = 1;
  Int3 direction = {1, 0, 0}; // +X direction

  auto recv_halo =
      halo::create_recv_halo<backend::CpuTag>(decomp, rank, direction, halo_width);

  REQUIRE(recv_halo.size() > 0);
  REQUIRE(recv_halo.is_sorted());

  // Verify indices are valid (spot check)
  auto indices = sparsevector::get_index(recv_halo);
  auto local_world = decomposition::get_subworld(decomp, rank);
  auto local_size = world::get_size(local_world);
  size_t local_total = static_cast<size_t>(local_size[0]) *
                       static_cast<size_t>(local_size[1]) *
                       static_cast<size_t>(local_size[2]);

  // Spot check: verify first, middle, and last indices are valid
  if (!indices.empty()) {
    REQUIRE(indices[0] < local_total);
    REQUIRE(indices[indices.size() / 2] < local_total);
    REQUIRE(indices[indices.size() - 1] < local_total);
  }
}

TEST_CASE("Send and recv halo have same size", "[halo][pattern]") {
  auto world = world::create(GridSize({64), PhysicalOrigin(64), GridSpacing(64}));
  auto decomp = decomposition::create(world, {2, 2, 1});

  int rank = 0;
  int halo_width = 1;
  Int3 direction = {1, 0, 0};

  auto send_halo =
      halo::create_send_halo<backend::CpuTag>(decomp, rank, direction, halo_width);
  auto recv_halo =
      halo::create_recv_halo<backend::CpuTag>(decomp, rank, direction, halo_width);

  // Send and receive halos should have the same number of elements
  REQUIRE(send_halo.size() == recv_halo.size());
}

TEST_CASE("Create halo patterns for all face neighbors", "[halo][pattern]") {
  auto world = world::create(GridSize({64), PhysicalOrigin(64), GridSpacing(64}));
  auto decomp = decomposition::create(world, {2, 2, 1});

  int rank = 0;
  int halo_width = 1;

  auto patterns = halo::create_halo_patterns<backend::CpuTag>(
      decomp, rank, halo::Connectivity::Faces, halo_width);

  // Should have neighbors in some directions (depends on rank position)
  // Rank 0 in 2×2×1 grid should have neighbors in +X and +Y directions
  REQUIRE(patterns.size() > 0);
  REQUIRE(patterns.size() <= 6); // Max 6 face neighbors

  // Verify each pattern has send and recv halos
  for (const auto &[direction, halos] : patterns) {
    const auto &[send_halo, recv_halo] = halos;
    REQUIRE(send_halo.size() == recv_halo.size());
    REQUIRE(send_halo.size() > 0);
  }
}

TEST_CASE("Gather from local field using send halo", "[halo][gather]") {
  auto world = world::create(GridSize({64), PhysicalOrigin(64), GridSpacing(64}));
  auto decomp = decomposition::create(world, {2, 2, 1});

  int rank = 0;
  int halo_width = 1;
  Int3 direction = {1, 0, 0};

  auto send_halo_indices =
      halo::create_send_halo<backend::CpuTag>(decomp, rank, direction, halo_width);

  // Create local field
  auto local_world = decomposition::get_subworld(decomp, rank);
  auto local_size = world::get_size(local_world);
  size_t local_total = static_cast<size_t>(local_size[0]) *
                       static_cast<size_t>(local_size[1]) *
                       static_cast<size_t>(local_size[2]);

  std::vector<double> local_field(local_total);
  // Fill with known pattern
  for (size_t i = 0; i < local_total; ++i) {
    local_field[i] = static_cast<double>(i);
  }

  // Create SparseVector for values (using same indices)
  auto indices_vec = sparsevector::get_index(send_halo_indices);
  core::SparseVector<backend::CpuTag, double> send_halo_values(indices_vec);

  // Gather values from local field
  gather(send_halo_values, local_field);

  // Verify gathered values
  auto gathered_data = sparsevector::get_data(send_halo_values);
  auto indices = sparsevector::get_index(send_halo_values);

  REQUIRE(gathered_data.size() == indices.size());
  // Spot check: verify first, middle, and last gathered values
  if (!gathered_data.empty()) {
    REQUIRE(gathered_data[0] == local_field[indices[0]]);
    size_t mid = gathered_data.size() / 2;
    REQUIRE(gathered_data[mid] == local_field[indices[mid]]);
    REQUIRE(gathered_data[gathered_data.size() - 1] ==
            local_field[indices[indices.size() - 1]]);
  }
}
