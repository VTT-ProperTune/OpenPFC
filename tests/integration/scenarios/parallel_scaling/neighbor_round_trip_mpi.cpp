// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <map>
#include <mpi.h>
#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

using namespace pfc;

TEST_CASE("4-rank neighbor_rank round-trip on non-cubic grid", "[MPI][parallel_scaling]") {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Skip if not exactly 4 ranks
  if (size != 4) {
    return;
  }

  // Construct non-cubic, non-periodic domain: 8x4x2 global extents
  const auto domain = Domain{Int3{8, 4, 2}, Real3{1.0, 1.0, 1.0}, Real3{0.0, 0.0, 0.0},
                             Bool3{false, false, false}};

  // Build 2x2x1 decomposition across 4 ranks
  const Int3 grid{2, 2, 1};
  auto decomp = decomposition::create(domain, grid);

  // Define expected neighbor pairs based on 2x2x1 topology on non-periodic grid:
  // Rank layout: rank 0 at (0,0,0), rank 1 at (1,0,0),
  //             rank 2 at (0,1,0), rank 3 at (1,1,0)
  // No neighbor exists at boundaries (-1)
  const std::array<Int3, 6> directions = {Int3{1, 0, 0},  Int3{-1, 0, 0},
                                           Int3{0, 1, 0},  Int3{0, -1, 0},
                                           Int3{0, 0, 1},  Int3{0, 0, -1}};

  std::map<int, std::map<Int3, int>> expected_neighbors = {
      {0, {{Int3{1, 0, 0}, 1}, {Int3{0, 1, 0}, 2}, {Int3{0, 0, 1}, -1},
           {Int3{-1, 0, 0}, -1}, {Int3{0, -1, 0}, -1}, {Int3{0, 0, -1}, -1}}},
      {1,
       {{Int3{1, 0, 0}, -1}, {Int3{0, 1, 0}, 3}, {Int3{0, 0, 1}, -1},
        {Int3{-1, 0, 0}, 0}, {Int3{0, -1, 0}, -1}, {Int3{0, 0, -1}, -1}}},
      {2,
       {{Int3{1, 0, 0}, 3}, {Int3{0, 1, 0}, -1}, {Int3{0, 0, 1}, -1},
        {Int3{-1, 0, 0}, -1}, {Int3{0, -1, 0}, 0}, {Int3{0, 0, -1}, -1}}},
      {3,
       {{Int3{1, 0, 0}, -1}, {Int3{0, 1, 0}, -1}, {Int3{0, 0, 1}, -1},
        {Int3{-1, 0, 0}, 2}, {Int3{0, -1, 0}, 1}, {Int3{0, 0, -1}, -1}}}};

  // Verify get_neighbor_rank returns expected neighbor for each direction
  for (const auto &direction : directions) {
    const int expected_neighbor = expected_neighbors[rank].at(direction);
    const int actual_neighbor = decomposition::get_neighbor_rank(decomp, rank, direction);
    REQUIRE(actual_neighbor == expected_neighbor);
  }

  // Verify round-trip: if rank A has neighbor B in direction D,
  // then rank B must have neighbor A in the opposite direction
  bool all_roundtrips_valid = true;
  for (const auto &direction : directions) {
    const int neighbor = expected_neighbors[rank].at(direction);

    if (neighbor != -1) {
      const Int3 opposite = Int3{-direction[0], -direction[1], -direction[2]};
      const int back_rank = decomposition::get_neighbor_rank(decomp, neighbor, opposite);
      all_roundtrips_valid &= (back_rank == rank);
    }
  }

  // Use MPI_Allreduce to ensure all ranks agree on round-trip validity
  int global_roundtrips_valid = 0;
  MPI_Allreduce(&all_roundtrips_valid, &global_roundtrips_valid, 1, MPI_C_BOOL, MPI_LAND,
                MPI_COMM_WORLD);
  REQUIRE(global_roundtrips_valid == 1);

  // Verify field_from_subdomain constructs valid pfc::data::Field
  const auto field = pfc::data::field_from_subdomain<double>(decomp, rank, /*halo=*/0);

  // Confirm field dimensions match local subdomain extents
  const auto local_box = decomposition::local_box(decomp, rank);
  const auto &field_domain = field.domain();
  
  REQUIRE(field_domain.size == domain.size);
  REQUIRE(field.box().size == local_box.size);
  REQUIRE(field.box().low == local_box.low);
  REQUIRE(field.box().high == local_box.high);
}
