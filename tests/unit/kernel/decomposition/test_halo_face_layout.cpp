// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/decomposition/halo_pattern.hpp>

using namespace pfc;

TEST_CASE("face_halo_counts matches create_recv_halo per direction", "[halo][layout]") {
  auto world = world::create(GridSize({64, 64, 64}));
  auto decomp = decomposition::create(world, {2, 2, 1});
  const int rank = 0;
  const int hw = 2;
  const std::array<Int3, 6> dirs = {{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0},
                                     {0, 0, 1}, {0, 0, -1}}};

  auto fc = halo::face_halo_counts(decomp, rank, hw);
  for (int i = 0; i < 6; ++i) {
    auto recv = halo::create_recv_halo<backend::CpuTag>(decomp, rank, dirs[i], hw);
    REQUIRE(fc.counts[static_cast<size_t>(i)] == recv.size());
  }
}

TEST_CASE("face_halo_counts_analytic matches pattern-based counts", "[halo][layout]") {
  auto world = world::create(GridSize({32, 48, 16}));
  auto decomp = decomposition::create(world, {2, 1, 1});
  const int rank = 0;
  const int hw = 1;
  auto local = world::get_size(decomposition::get_subworld(decomp, rank));
  auto fc = halo::face_halo_counts(decomp, rank, hw);
  auto an = halo::face_halo_counts_analytic(local[0], local[1], local[2], hw);
  for (int i = 0; i < 6; ++i) {
    REQUIRE(fc.counts[static_cast<size_t>(i)] == an.counts[static_cast<size_t>(i)]);
  }
}

TEST_CASE("allocate_face_halos sizes", "[halo][layout]") {
  auto world = world::create(GridSize({16, 16, 16}));
  auto decomp = decomposition::create(world, {2, 2, 2});
  const int rank = 0;
  const int hw = 1;
  auto bufs = halo::allocate_face_halos<double>(decomp, rank, hw);
  auto fc = halo::face_halo_counts(decomp, rank, hw);
  for (int i = 0; i < 6; ++i) {
    REQUIRE(bufs[static_cast<size_t>(i)].size() == fc.counts[static_cast<size_t>(i)]);
  }
}
