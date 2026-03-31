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
    for (const auto &entry : faces) {
      REQUIRE(entry.second ==
              decomposition::get_neighbor_rank(decomp, rank, entry.first));
    }

    auto alln = decomposition::find_all_neighbors(decomp, rank);
    REQUIRE(alln.size() == 26);
    for (const auto &entry : alln) {
      REQUIRE(entry.second ==
              decomposition::get_neighbor_rank(decomp, rank, entry.first));
    }
  }
}
