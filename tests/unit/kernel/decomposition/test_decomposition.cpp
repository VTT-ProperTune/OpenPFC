// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>

using namespace Catch::Matchers;
using namespace pfc;
using namespace pfc::types;

TEST_CASE("Decomposition - basic functionality", "[decomposition][unit]") {
  // Create a dummy World object for testing
  auto world = world::create(GridSize({128, 128, 128}));

  SECTION("Construction and getters") {
    auto decomposition = decomposition::create(world, 1);

    REQUIRE(get_world(decomposition) == world);
  }

  SECTION("make_decomposition(world, rank, num_domains) matches domain count") {
    auto decomposition = make_decomposition(world, 0, 8);

    REQUIRE(decomposition::get_num_domains(decomposition) == 8);
    REQUIRE(decomposition::get_world(decomposition) == world);
  }
}

TEST_CASE("Decomposition - periodic neighbor ranks", "[decomposition][unit]") {
  auto world = world::create(GridSize({16, 16, 16}));
  const Int3 grid{2, 2, 2};
  auto decomp = decomposition::create(world, grid);

  REQUIRE(decomposition::get_num_domains(decomp) == 8);

  SECTION("get_neighbor_rank returns -1 for out-of-range rank") {
    REQUIRE(decomposition::get_neighbor_rank(decomp, -1, Int3{1, 0, 0}) == -1);
    REQUIRE(decomposition::get_neighbor_rank(decomp, 8, Int3{1, 0, 0}) == -1);
  }

  SECTION("get_neighbor_rank torus layout for rank 0 on 2x2x2 grid") {
    // Rank index: z * 4 + y * 2 + x with (x,y,z) in [0,1]^3
    REQUIRE(decomposition::get_neighbor_rank(decomp, 0, Int3{1, 0, 0}) == 1);
    REQUIRE(decomposition::get_neighbor_rank(decomp, 0, Int3{0, 1, 0}) == 2);
    REQUIRE(decomposition::get_neighbor_rank(decomp, 0, Int3{0, 0, 1}) == 4);
    REQUIRE(decomposition::get_neighbor_rank(decomp, 0, Int3{-1, 0, 0}) == 1);
    REQUIRE(decomposition::get_neighbor_rank(decomp, 0, Int3{0, -1, 0}) == 2);
    REQUIRE(decomposition::get_neighbor_rank(decomp, 0, Int3{0, 0, -1}) == 4);
  }

  SECTION("find_face_neighbors and find_all_neighbors match get_neighbor_rank") {
    const int rank = 0;
    auto faces = decomposition::find_face_neighbors(decomp, rank);
    REQUIRE(faces.size() == 6);
    bool neighbors_match = true;
    for (const auto &entry : faces) {
      neighbors_match &= entry.second ==
                         decomposition::get_neighbor_rank(decomp, rank, entry.first);
    }

    auto alln = decomposition::find_all_neighbors(decomp, rank);
    REQUIRE(alln.size() == 26);
    for (const auto &entry : alln) {
      neighbors_match &= entry.second ==
                         decomposition::get_neighbor_rank(decomp, rank, entry.first);
    }
    REQUIRE(neighbors_match);
  }
}

TEST_CASE("test_create_rejects_excessive_nparts", "[decomposition][unit][error]") {
  using namespace pfc;
  // World: 2x2x2, only 8 total grid points
  // Requesting 9 decompositions exceeds what HeFFTe can partition
  auto world = world::create(GridSize({2, 2, 2}));
  REQUIRE_THROWS_AS(::decomposition::create(world, 9), std::invalid_argument);
}

TEST_CASE("test_create_rejects_invalid_grid", "[decomposition][unit][error]") {
  using namespace pfc;
  auto world = world::create(GridSize({128, 128, 128}));
  // Grid dimension of zero is invalid
  REQUIRE_THROWS_AS(::decomposition::create(world, Int3{300, 1, 1}),
                    std::invalid_argument);
}

TEST_CASE("Decomposition - split_world box ordering matches x-fastest ranks",
          "[decomposition][unit]") {
  using namespace pfc;
  // Construction runs validate_split_world_ordering (audit 4.9). If HeFFTe ever
  // enumerated boxes in a different order, construction would throw here.
  // Non-cubic grids exercise gx != gy != gz.
  for (const Int3 grid :
       {Int3{2, 3, 4}, Int3{1, 2, 4}, Int3{4, 1, 2}, Int3{3, 3, 1}}) {
    auto world = world::create(GridSize({24, 24, 24}));
    REQUIRE_NOTHROW(decomposition::create(world, grid));
  }
}

TEST_CASE("Decomposition - get_neighbor_rank round-trips on non-cubic grids",
          "[decomposition][unit]") {
  using namespace pfc;
  const Int3 grid{2, 3, 4};
  auto world = world::create(GridSize({24, 24, 24}));
  auto decomp = decomposition::create(world, grid);
  const int n = decomposition::get_num_domains(decomp);
  REQUIRE(n == grid[0] * grid[1] * grid[2]);

  const std::array<Int3, 6> dirs = {Int3{1, 0, 0},  Int3{-1, 0, 0}, Int3{0, 1, 0},
                                    Int3{0, -1, 0}, Int3{0, 0, 1},  Int3{0, 0, -1}};
  bool roundtrips = true;
  for (int r = 0; r < n; ++r) {
    for (const Int3 &d : dirs) {
      const int nb = decomposition::get_neighbor_rank(decomp, r, d);
      const Int3 back{-d[0], -d[1], -d[2]};
      // Stepping back from the neighbor along the opposite direction returns r.
      roundtrips &= (decomposition::get_neighbor_rank(decomp, nb, back) == r);
    }
  }
  REQUIRE(roundtrips);
}

TEST_CASE("Decomposition - per-axis periodicity in get_neighbor_rank (M1.3)",
          "[decomposition][unit]") {
  using namespace pfc;
  // Non-periodic in x, periodic in y and z.
  auto world =
      world::from_bounds({16, 16, 16}, {0, 0, 0}, {16, 16, 16}, {false, true, true});
  auto decomp = decomposition::create(world, Int3{2, 2, 2});
  REQUIRE(decomposition::get_num_domains(decomp) == 8);

  SECTION("non-periodic x boundary has no neighbor") {
    // rank 0 sits at x-coord 0: stepping to -x leaves the domain -> no neighbor.
    REQUIRE(decomposition::get_neighbor_rank(decomp, 0, Int3{-1, 0, 0}) == -1);
    // rank 1 sits at x-coord 1 (the max): stepping to +x leaves the domain.
    REQUIRE(decomposition::get_neighbor_rank(decomp, 1, Int3{1, 0, 0}) == -1);
    // Interior x step stays valid.
    REQUIRE(decomposition::get_neighbor_rank(decomp, 0, Int3{1, 0, 0}) == 1);
  }

  SECTION("periodic y and z still wrap") {
    REQUIRE(decomposition::get_neighbor_rank(decomp, 0, Int3{0, -1, 0}) == 2);
    REQUIRE(decomposition::get_neighbor_rank(decomp, 0, Int3{0, 0, -1}) == 4);
  }

  SECTION("find_face_neighbors drops the non-periodic face for a boundary rank") {
    // rank 0 loses its -x face; +x (rank 1), ±y, ±z remain valid -> 5.
    auto faces = decomposition::find_face_neighbors(decomp, 0);
    REQUIRE(faces.size() == 6);
    int valid = 0;
    for (const auto &e : faces)
      if (e.second >= 0) ++valid;
    REQUIRE(valid == 5);
  }
}
