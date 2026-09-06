// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdlib>
#include <heffte.h>
#include <vector>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/decomposition/brick_split.hpp>

using namespace pfc;
using pfc::decomposition::min_surface_proc_grid;
using pfc::decomposition::node_aware_fft_proc_grid;
using pfc::decomposition::slab_proc_grid;
using pfc::decomposition::spectral_fft_proc_grid;
using pfc::decomposition::split_box;

namespace {

// HeFFTe 2.4.1 `proc_setup_min_surface` uses `j_max % j` with
// `j_max = min(num_procs/i, size_y)`. That misses some valid grids and
// `assert`s when it finds none (CUDA/Debug HeFFTe). NDEBUG HeFFTe then
// returns {1,1,1}. Our splitter copies the same loop so FFT inbox geometry
// stays identical; skip the live HeFFTe call when that loop finds nothing.
bool heffte_loop_finds_grid(const Int3 &size, int num_procs) {
  if (num_procs == 1) {
    return true;
  }
  const int i_max = std::min(num_procs, size[0]);
  for (int i = 1; i <= i_max; ++i) {
    if (num_procs % i != 0) {
      continue;
    }
    const int j_max = std::min(num_procs / i, size[1]);
    for (int j = 1; j <= j_max; ++j) {
      if (j_max % j != 0) {
        continue;
      }
      const int k = num_procs / (i * j);
      if (k >= 1 && k <= size[2] && i * j * k == num_procs) {
        return true;
      }
    }
  }
  return false;
}

void require_matches_heffte(const Int3 &size, int nparts) {
  const heffte::box3d<int> world({0, 0, 0}, {size[0] - 1, size[1] - 1, size[2] - 1});
  const auto href = heffte::proc_setup_min_surface(world, nparts);
  const Int3 grid = min_surface_proc_grid(size, nparts);
  REQUIRE(grid[0] == href[0]);
  REQUIRE(grid[1] == href[1]);
  REQUIRE(grid[2] == href[2]);

  const Box3i box =
      Box3i::from_bounds({0, 0, 0}, {size[0] - 1, size[1] - 1, size[2] - 1});
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
    const long long ncells = static_cast<long long>(sz[0]) * sz[1] * sz[2];
    for (int np : ranks) {
      if (np > ncells) {
        continue;
      }
      if (!heffte_loop_finds_grid(sz, np)) {
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

TEST_CASE("slab_proc_grid is 1D along a divisible axis", "[brick_split][unit]") {
  const Int3 cube{768, 768, 768};
  REQUIRE(slab_proc_grid(cube, 1) == Int3{1, 1, 1});
  REQUIRE(slab_proc_grid(cube, 8) == Int3{1, 1, 8});
  REQUIRE(slab_proc_grid(cube, 16) == Int3{1, 1, 16});
  REQUIRE(slab_proc_grid(cube, 24) == Int3{1, 1, 24});
  REQUIRE(slab_proc_grid(cube, 32) == Int3{1, 1, 32});
  const Int3 xy{16, 8, 7};
  REQUIRE(slab_proc_grid(xy, 8) == Int3{1, 8, 1});
}

TEST_CASE("slab_proc_grid keeps the r2c axis in-plane", "[brick_split][unit]") {
  const Int3 cube{768, 768, 768};
  REQUIRE(slab_proc_grid(cube, 16, 0) == Int3{1, 1, 16});
  REQUIRE(slab_proc_grid(cube, 16, 1) == Int3{1, 1, 16});
  REQUIRE(slab_proc_grid(cube, 16, 2) == Int3{1, 16, 1});
  const Int3 tall{64, 32, 16};
  REQUIRE(slab_proc_grid(tall, 16, 0) == Int3{1, 1, 16});
  REQUIRE(slab_proc_grid(tall, 16, 2) == Int3{1, 16, 1});
  const Int3 flat{64, 8, 32};
  REQUIRE(slab_proc_grid(flat, 16, 2) == Int3{16, 1, 1});
}

TEST_CASE("spectral_fft_proc_grid keeps bricks on one node", "[brick_split][unit]") {
  const Int3 cube{768, 768, 768};
  REQUIRE(spectral_fft_proc_grid(cube, 8) == min_surface_proc_grid(cube, 8));
}

TEST_CASE("node_aware_fft_proc_grid is 1x8xnnodes off-node", "[brick_split][unit]") {
  const Int3 cube{768, 768, 768};
  REQUIRE(node_aware_fft_proc_grid(cube, 8) == Int3{0, 0, 0});
  REQUIRE(node_aware_fft_proc_grid(cube, 16) == Int3{1, 8, 2});
  REQUIRE(node_aware_fft_proc_grid(cube, 24) == Int3{1, 8, 3});
  REQUIRE(node_aware_fft_proc_grid(cube, 32) == Int3{1, 8, 4});
  REQUIRE(spectral_fft_proc_grid(cube, 16) == Int3{1, 1, 16});
  REQUIRE(spectral_fft_proc_grid(cube, 24) == Int3{1, 1, 24});
  REQUIRE(spectral_fft_proc_grid(cube, 32) == Int3{1, 1, 32});
  struct Clear {
    ~Clear() { unsetenv("OPENPFC_FFT_NODE_GRID"); }
  } clear;
  REQUIRE(setenv("OPENPFC_FFT_NODE_GRID", "1", 1) == 0);
  REQUIRE(spectral_fft_proc_grid(cube, 16) == Int3{1, 8, 2});
  REQUIRE(spectral_fft_proc_grid(cube, 24) == Int3{1, 8, 3});
  REQUIRE(unsetenv("OPENPFC_FFT_NODE_GRID") == 0);
  REQUIRE(spectral_fft_proc_grid(cube, 16) == Int3{1, 1, 16});

  const Box3i world = Box3i::from_bounds({0, 0, 0}, {767, 767, 767});
  const auto boxes = split_box(world, Int3{1, 8, 2});
  REQUIRE(boxes.size() == 16);
  // x-fastest: ranks 0-7 are z-half 0 (one node), 8-15 are z-half 1.
  REQUIRE(boxes[0].low[2] == 0);
  REQUIRE(boxes[7].high[2] == 383);
  REQUIRE(boxes[8].low[2] == 384);
  REQUIRE(boxes[15].high[2] == 767);
  REQUIRE(boxes[0].low[1] == 0);
  REQUIRE(boxes[1].low[1] > boxes[0].low[1]);
}

TEST_CASE("spectral_fft_proc_grid honors OPENPFC_FFT_PROC_GRID",
          "[brick_split][unit]") {
  struct Clear {
    ~Clear() { unsetenv("OPENPFC_FFT_PROC_GRID"); }
  } clear;
  const Int3 cube{768, 768, 768};
  REQUIRE(unsetenv("OPENPFC_FFT_PROC_GRID") == 0);
  REQUIRE(pfc::decomposition::fft_proc_grid_override() == Int3{0, 0, 0});
  REQUIRE(setenv("OPENPFC_FFT_PROC_GRID", "1,2,8", 1) == 0);
  REQUIRE(spectral_fft_proc_grid(cube, 16) == Int3{1, 2, 8});
  REQUIRE(setenv("OPENPFC_FFT_PROC_GRID", "1x4x4", 1) == 0);
  REQUIRE(spectral_fft_proc_grid(cube, 16) == Int3{1, 4, 4});
  REQUIRE(setenv("OPENPFC_FFT_PROC_GRID", "2,8,1", 1) == 0);
  REQUIRE(spectral_fft_proc_grid(cube, 16) == Int3{2, 8, 1});
  REQUIRE(setenv("OPENPFC_FFT_PROC_GRID", "1,1,15", 1) == 0); // not 16 ranks
  REQUIRE(spectral_fft_proc_grid(cube, 16) == Int3{1, 1, 16});
  REQUIRE(unsetenv("OPENPFC_FFT_PROC_GRID") == 0);
}

TEST_CASE("slab_proc_grid honors OPENPFC_FFT_SLAB_AXIS", "[brick_split][unit]") {
  struct Clear {
    ~Clear() { unsetenv("OPENPFC_FFT_SLAB_AXIS"); }
  } clear;
  const Int3 cube{768, 768, 768};
  REQUIRE(unsetenv("OPENPFC_FFT_SLAB_AXIS") == 0);
  REQUIRE(pfc::decomposition::fft_slab_axis_override() == -1);
  REQUIRE(slab_proc_grid(cube, 16) == Int3{1, 1, 16});
  REQUIRE(setenv("OPENPFC_FFT_SLAB_AXIS", "y", 1) == 0);
  REQUIRE(pfc::decomposition::fft_slab_axis_override() == 1);
  REQUIRE(slab_proc_grid(cube, 16) == Int3{1, 16, 1});
  REQUIRE(spectral_fft_proc_grid(cube, 16) == Int3{1, 16, 1});
  REQUIRE(setenv("OPENPFC_FFT_SLAB_AXIS", "x", 1) == 0);
  REQUIRE(slab_proc_grid(cube, 16) == Int3{16, 1, 1});
  REQUIRE(setenv("OPENPFC_FFT_SLAB_AXIS", "z", 1) == 0);
  REQUIRE(slab_proc_grid(cube, 16) == Int3{1, 1, 16});
  REQUIRE(unsetenv("OPENPFC_FFT_SLAB_AXIS") == 0);
  REQUIRE(slab_proc_grid(cube, 16) == Int3{1, 1, 16});
}
