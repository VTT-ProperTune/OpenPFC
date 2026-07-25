// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

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
  auto world = world::create(GridSize({16, 16, 16}));
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
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {-1, 0, 0}) == -1);

    // Rank 1 at (1,0,0): +X direction crosses non-periodic boundary
    rank = 1;
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {1, 0, 0}) == -1);

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
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {0, -1, 0}) == -1);

    // Rank 2 at (0,1,0): +Y direction crosses non-periodic boundary
    rank = 2;
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {0, 1, 0}) == -1);

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
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {0, 0, -1}) == -1);

    // Rank 4 at (0,0,1): +Z direction crosses non-periodic boundary
    rank = 4;
    REQUIRE(decomposition::get_neighbor_rank(decomp, domain, rank, {0, 0, 1}) == -1);

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
}
