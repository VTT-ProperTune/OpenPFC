// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <mpi.h>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>

using namespace pfc;
using namespace pfc::types;

TEST_CASE("get_neighbor_rank with per-axis periodicity",
          "[decomposition][neighbors][unit]") {
  // Create a 2x2x2 decomposition for testing
  auto world = world::create(GridSize({16, 16, 16}).to_vector3());
  const Int3 grid{2, 2, 2};
  auto decomp = decomposition::create(world, grid);

  SECTION("fully periodic domain returns valid neighbors for all directions") {
    auto domain = decomposition::domain(decomp);
    // Default domain is fully periodic
    REQUIRE(pfc::domain::is_periodic(domain, 0) == true);
    REQUIRE(pfc::domain::is_periodic(domain, 1) == true);
    REQUIRE(pfc::domain::is_periodic(domain, 2) == true);

    // All directions should return valid neighbors (wraps around)
    int rank = 0;
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {1, 0, 0}) == 1);
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {-1, 0, 0}) == 1);
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {0, 1, 0}) == 2);
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {0, -1, 0}) == 2);
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {0, 0, 1}) == 4);
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {0, 0, -1}) == 4);
  }

  SECTION("non-periodic X axis returns -1 for X boundary crossing") {
    // Create a domain with non-periodic X axis
    auto domain = pfc::domain::create(
        GridSize({16, 16, 16}), PhysicalOrigin({0.0, 0.0, 0.0}),
        GridSpacing({1.0, 1.0, 1.0}), Bool3{false, true, true});

    REQUIRE(pfc::domain::is_periodic(domain, 0) == false); // X is non-periodic
    REQUIRE(pfc::domain::is_periodic(domain, 1) == true);  // Y is periodic
    REQUIRE(pfc::domain::is_periodic(domain, 2) == true);  // Z is periodic

    // Rank 0 at (0,0,0): -X direction crosses non-periodic boundary
    int rank = 0;
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {-1, 0, 0}) == MPI_PROC_NULL);

    // Rank 1 at (1,0,0): +X direction crosses non-periodic boundary
    rank = 1;
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {1, 0, 0}) == MPI_PROC_NULL);

    // Y and Z directions should still wrap (periodic)
    rank = 0;
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {0, 1, 0}) == 2);
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {0, 0, 1}) == 4);
  }

  SECTION("non-periodic Y axis returns -1 for Y boundary crossing") {
    // Create a domain with non-periodic Y axis
    auto domain = pfc::domain::create(
        GridSize({16, 16, 16}), PhysicalOrigin({0.0, 0.0, 0.0}),
        GridSpacing({1.0, 1.0, 1.0}), Bool3{true, false, true});

    REQUIRE(pfc::domain::is_periodic(domain, 0) == true);  // X is periodic
    REQUIRE(pfc::domain::is_periodic(domain, 1) == false); // Y is non-periodic
    REQUIRE(pfc::domain::is_periodic(domain, 2) == true);  // Z is periodic

    // Rank 0 at (0,0,0): -Y direction crosses non-periodic boundary
    int rank = 0;
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {0, -1, 0}) == MPI_PROC_NULL);

    // Rank 2 at (0,1,0): +Y direction crosses non-periodic boundary
    rank = 2;
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {0, 1, 0}) == MPI_PROC_NULL);

    // X and Z directions should still wrap (periodic)
    rank = 0;
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {1, 0, 0}) == 1);
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {0, 0, 1}) == 4);
  }

  SECTION("non-periodic Z axis returns -1 for Z boundary crossing") {
    // Create a domain with non-periodic Z axis
    auto domain = pfc::domain::create(
        GridSize({16, 16, 16}), PhysicalOrigin({0.0, 0.0, 0.0}),
        GridSpacing({1.0, 1.0, 1.0}), Bool3{true, true, false});

    REQUIRE(pfc::domain::is_periodic(domain, 0) == true);  // X is periodic
    REQUIRE(pfc::domain::is_periodic(domain, 1) == true);  // Y is periodic
    REQUIRE(pfc::domain::is_periodic(domain, 2) == false); // Z is non-periodic

    // Rank 0 at (0,0,0): -Z direction crosses non-periodic boundary
    int rank = 0;
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {0, 0, -1}) == MPI_PROC_NULL);

    // Rank 4 at (0,0,1): +Z direction crosses non-periodic boundary
    rank = 4;
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {0, 0, 1}) == MPI_PROC_NULL);

    // X and Y directions should still wrap (periodic)
    rank = 0;
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {1, 0, 0}) == 1);
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {0, 1, 0}) == 2);
  }

  SECTION("backward-compatible overload works with default periodicity") {
    // The overload without explicit Domain should work as before
    int rank = 0;
    REQUIRE(decomposition::get_neighbor_rank(decomp, rank, {1, 0, 0}) == 1);
    REQUIRE(decomposition::get_neighbor_rank(decomp, rank, {0, 1, 0}) == 2);
    REQUIRE(decomposition::get_neighbor_rank(decomp, rank, {0, 0, 1}) == 4);
    REQUIRE(decomposition::get_neighbor_rank(decomp, rank, {-1, 0, 0}) == 1);
    REQUIRE(decomposition::get_neighbor_rank(decomp, rank, {0, -1, 0}) == 2);
    REQUIRE(decomposition::get_neighbor_rank(decomp, rank, {0, 0, -1}) == 4);
  }

  SECTION("4-rank 2x2 mixed periodicity round-trip") {
    // Create a 2x2x1 decomposition (4 ranks) for round-trip testing
    auto world_2x2 = world::create(GridSize({16, 16, 1}).to_vector3());
    const Int3 grid_2x2{2, 2, 1};
    auto decomp_2x2 = decomposition::create(world_2x2, grid_2x2);

    // Domain with X periodic, Y non-periodic, Z non-periodic
    auto mixed_domain = pfc::domain::create(
        GridSize({16, 16, 1}), PhysicalOrigin({0.0, 0.0, 0.0}),
        GridSpacing({1.0, 1.0, 1.0}), Bool3{true, false, false});

    // Rank layout for 2x2x1 grid:
    // Rank 0: (x=0, y=0, z=0)  Rank 1: (x=1, y=0, z=0)
    // Rank 2: (x=0, y=1, z=0)  Rank 3: (x=1, y=1, z=0)

    const std::array<Int3, 4> dirs = {Int3{1, 0, 0},  Int3{-1, 0, 0},
                                        Int3{0, 1, 0},  Int3{0, -1, 0}};

    // Test round-trips for all ranks with mixed periodicity
    for (int r = 0; r < 4; ++r) {
      for (const Int3 &d : dirs) {
        int nb = decomposition::get_neighbor_rank(decomp_2x2, mixed_domain, r, d);

        if (nb != MPI_PROC_NULL) {
          const Int3 back{-d[0], -d[1], -d[2]};
          int back_rank = decomposition::get_neighbor_rank(decomp_2x2, mixed_domain, nb, back);
          REQUIRE(back_rank == r); // Round-trip should work for valid neighbors
        }
      }
    }

    // Specific checks for boundary cases
    // Rank 0 at (0,0,0): -X wraps to rank 1 (periodic), -Y returns MPI_PROC_NULL (non-periodic)
    REQUIRE(decomposition::get_neighbor_rank(decomp_2x2, mixed_domain, 0, {-1, 0, 0}) == 1);
    REQUIRE(decomposition::get_neighbor_rank(decomp_2x2, mixed_domain, 0, {0, -1, 0}) == MPI_PROC_NULL);
    // Rank 2 at (0,1,0): +X returns rank 3, +Y returns MPI_PROC_NULL (non-periodic)
    REQUIRE(decomposition::get_neighbor_rank(decomp_2x2, mixed_domain, 2, {1, 0, 0}) == 3);
    REQUIRE(decomposition::get_neighbor_rank(decomp_2x2, mixed_domain, 2, {0, 1, 0}) == MPI_PROC_NULL);
  }
}

TEST_CASE("test_default_all_periodic", "[decomposition][neighbors][unit]") {
  // Test that when Domain periodicity is not explicitly configured,
  // all three axes (X, Y, Z) are periodic by default
  // and get_neighbor_rank returns valid neighbor ranks for all six directions

  // Create a 2x2x2 decomposition
  auto world = world::create(GridSize({16, 16, 16}).to_vector3());
  const Int3 grid{2, 2, 2};
  auto decomp = decomposition::create(world, grid);

  // Get the domain (should be fully periodic by default)
  auto domain = decomposition::domain(decomp);

  // Verify that all axes are periodic by default
  REQUIRE(pfc::domain::is_periodic(domain, 0) == true);
  REQUIRE(pfc::domain::is_periodic(domain, 1) == true);
  REQUIRE(pfc::domain::is_periodic(domain, 2) == true);

  // Test that get_neighbor_rank returns valid ranks for all six coordinate directions
  // on rank 0 (at position (0,0,0) in the grid)
  int rank = 0;
  REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {1, 0, 0}) == 1);   // +X
  REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {-1, 0, 0}) == 1);  // -X (wraps)
  REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {0, 1, 0}) == 2);   // +Y
  REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {0, -1, 0}) == 2);  // -Y (wraps)
  REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {0, 0, 1}) == 4);   // +Z
  REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {0, 0, -1}) == 4);  // -Z (wraps)
}

TEST_CASE("test_mixed_periodicity_x_periodic_yz_nonperiodic", "[decomposition][neighbors][unit]") {
  // Test that get_neighbor_rank(rank, {+1,0,0}) returns the neighbor process on the +X edge
  // but get_neighbor_rank(rank, {0,+1,0}) and get_neighbor_rank(rank, {0,0,+1})
  // return MPI_PROC_NULL on +Y and +Z edges when X is periodic and Y, Z are non-periodic

  // Create a 2x2x2 decomposition
  auto world = world::create(GridSize({16, 16, 16}).to_vector3());
  const Int3 grid{2, 2, 2};
  auto decomp = decomposition::create(world, grid);

  // Create a domain with X periodic, Y and Z non-periodic
  auto mixed_domain = pfc::domain::create(
      GridSize({16, 16, 16}), PhysicalOrigin({0.0, 0.0, 0.0}),
      GridSpacing({1.0, 1.0, 1.0}), Bool3{true, false, false});

  // Verify periodicity configuration
  REQUIRE(pfc::domain::is_periodic(mixed_domain, 0) == true);  // X is periodic
  REQUIRE(pfc::domain::is_periodic(mixed_domain, 1) == false); // Y is non-periodic
  REQUIRE(pfc::domain::is_periodic(mixed_domain, 2) == false); // Z is non-periodic

  // Test rank 0 at (0,0,0): -X wraps to neighbor due to X periodicity, while
  // -Y and -Z return MPI_PROC_NULL (non-periodic boundaries)
  REQUIRE(decomposition::get_neighbor_rank(decomp, mixed_domain, 0, {-1, 0, 0}) != MPI_PROC_NULL); // -X wraps (periodic)
  REQUIRE(decomposition::get_neighbor_rank(decomp, mixed_domain, 0, {0, -1, 0}) == MPI_PROC_NULL); // -Y boundary (non-periodic)
  REQUIRE(decomposition::get_neighbor_rank(decomp, mixed_domain, 0, {0, 0, -1}) == MPI_PROC_NULL); // -Z boundary (non-periodic)
}
