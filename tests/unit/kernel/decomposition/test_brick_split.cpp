// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <heffte.h>
#include <vector>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/decomposition/brick_split.hpp>

using namespace pfc;
using pfc::decomposition::min_surface_proc_grid;
using pfc::decomposition::split_box;

namespace {

void require_matches_heffte(const Int3 &size, int nparts) {
  const heffte::box3d<int> world({0, 0, 0},
                                 {size[0] - 1, size[1] - 1, size[2] - 1});
  const auto href = heffte::proc_setup_min_surface(world, nparts);
  const Int3 grid = min_surface_proc_grid(size, nparts);
  REQUIRE(grid[0] == href[0]);
  REQUIRE(grid[1] == href[1]);
  REQUIRE(grid[2] == href[2]);

  const Box3i box = Box3i::from_bounds({0, 0, 0}, {size[0] - 1, size[1] - 1, size[2] - 1});
  const auto ours = split_box(box, grid);
  const auto theirs = heffte::split_world(world, href);
  REQUIRE(ours.size() == theirs.size());
  for (std::size_t i = 0; i < ours.size(); ++i) {
    REQUIRE(ours[i].low[0] == theirs[i].low[0]);
    REQUIRE(ours[i].low[1] == theirs[i].low[1]);
    REQUIRE(ours[i].low[2] == theirs[i].low[2]);
    REQUIRE(ours[i].high[0] == theirs[i].high[0]);
    REQUIRE(ours[i].high[1] == theirs[i].high[1]);
    REQUIRE(ours[i].high[2] == theirs[i].high[2]);
  }
}

} // namespace

TEST_CASE("min_surface_proc_grid and split_box match HeFFTe",
          "[brick_split][unit]") {
  const std::array<Int3, 6> sizes{{
      Int3{8, 8, 8},
      Int3{16, 8, 4},
      Int3{32, 32, 1},
      Int3{7, 5, 3},
      Int3{64, 64, 64},
      Int3{128, 16, 8},
  }};
  const int ranks[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32};
  int compared = 0;
  for (const auto &sz : sizes) {
    const long long ncells =
        static_cast<long long>(sz[0]) * sz[1] * sz[2];
    for (int np : ranks) {
      if (np > ncells) {
        continue;
      }
      require_matches_heffte(sz, np);
      ++compared;
    }
  }
  REQUIRE(compared >= 12);
}

TEST_CASE("split_box is x-fastest for an explicit 2x2x1 grid",
          "[brick_split][unit]") {
  const Box3i world = Box3i::from_bounds({0, 0, 0}, {7, 7, 0});
  const auto boxes = split_box(world, Int3{2, 2, 1});
  REQUIRE(boxes.size() == 4);
  REQUIRE(boxes[0].low[0] == 0);
  REQUIRE(boxes[1].low[0] > boxes[0].low[0]);
  REQUIRE(boxes[2].low[1] > boxes[0].low[1]);
}
