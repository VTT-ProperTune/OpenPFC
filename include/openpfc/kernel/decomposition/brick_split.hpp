// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file brick_split.hpp
 * @brief In-repo min-surface process grid and x-fastest brick split (ADR 0007).
 *
 * Matches `heffte::proc_setup_min_surface` / `heffte::split_world` so FFT
 * inbox geometry stays unchanged when Decomposition no longer calls HeFFTe.
 */

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/types.hpp>

namespace pfc::decomposition {

/// Process-grid factors of `num_procs` that minimise approximate box surface.
[[nodiscard]] inline Int3 min_surface_proc_grid(const Int3 &size, int num_procs) {
  if (num_procs <= 0) {
    throw std::invalid_argument("min_surface_proc_grid: num_procs must be > 0");
  }
  for (int d = 0; d < 3; ++d) {
    if (size[d] < 1) {
      throw std::invalid_argument("min_surface_proc_grid: size must be >= 1");
    }
  }
  if (num_procs == 1) {
    return Int3{1, 1, 1};
  }

  const std::array<long long, 3> all{size[0], size[1], size[2]};
  auto surface = [&](int i, int j, int k) -> long long {
    const long long bx = all[0] / i;
    const long long by = all[1] / j;
    const long long bz = all[2] / k;
    return bx * by + by * bz + bz * bx;
  };

  Int3 best{1, 1, 1};
  long long best_surface = std::numeric_limits<long long>::max();
  bool found = false;

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
      if (k < 1 || k > size[2] || i * j * k != num_procs) {
        continue;
      }
      const long long s = surface(i, j, k);
      if (s < best_surface) {
        best_surface = s;
        best = Int3{i, j, k};
        found = true;
      }
    }
  }
  if (!found) {
    throw std::invalid_argument(
        "min_surface_proc_grid: no " + std::to_string(size[0]) + "x" +
        std::to_string(size[1]) + "x" + std::to_string(size[2]) +
        " process grid for " + std::to_string(num_procs) + " ranks");
  }
  return best;
}

/// Off-node FFT. One LUMI-G node is 8 GCDs. A 1D slab (1×1×N) makes every
/// HeFFTe z-transpose an N-rank all-to-all. A 2D grid 1×8×nnodes keeps
/// consecutive ranks on one node so the y-pencil reshape is intra-node and
/// the z-pencil reshape is pairwise between nodes. Pair with HeFFTe pencils.
inline constexpr int kSpectralSlabMinRanks = 9;
inline constexpr int kSpectralNodeGcds = 8;

/// `OPENPFC_FFT_SLAB_AXIS` = `x`/`y`/`z` or `0`/`1`/`2`; -1 means unset.
[[nodiscard]] inline int fft_slab_axis_override() {
  const char *e = std::getenv("OPENPFC_FFT_SLAB_AXIS");
  if (e == nullptr || e[0] == '\0' || e[1] != '\0') {
    return -1;
  }
  if (e[0] == 'x' || e[0] == '0') {
    return 0;
  }
  if (e[0] == 'y' || e[0] == '1') {
    return 1;
  }
  if (e[0] == 'z' || e[0] == '2') {
    return 2;
  }
  return -1;
}

/**
 * @brief 1D process grid along the first axis that `num_procs` divides.
 *
 * Prefers axes that keep @p r2c_direction in-plane (default x, so z then y
 * then x). Splitting the r2c axis last avoids an extra HeFFTe reshape of the
 * reduced complex dimension. `OPENPFC_FFT_SLAB_AXIS` forces x, y, or z when
 * that axis divides `num_procs`. Falls back to @ref min_surface_proc_grid
 * when no axis divides evenly.
 */
[[nodiscard]] inline Int3 slab_proc_grid(const Int3 &size, int num_procs,
                                         int r2c_direction = 0) {
  if (num_procs <= 1) {
    return Int3{1, 1, 1};
  }
  if (r2c_direction < 0 || r2c_direction > 2) {
    r2c_direction = 0;
  }
  auto try_axis = [&](int d) -> Int3 {
    Int3 g{1, 1, 1};
    g[d] = num_procs;
    return g;
  };
  const int forced = fft_slab_axis_override();
  if (forced >= 0 && num_procs <= size[forced] && size[forced] % num_procs == 0) {
    return try_axis(forced);
  }
  // Last spatial axis first among the two that are not the r2c direction.
  static constexpr int kPref[3][3] = {{2, 1, 0}, {2, 0, 1}, {1, 0, 2}};
  for (int k = 0; k < 3; ++k) {
    const int d = kPref[r2c_direction][k];
    if (num_procs <= size[d] && size[d] % num_procs == 0) {
      return try_axis(d);
    }
  }
  return min_surface_proc_grid(size, num_procs);
}

/**
 * @brief 2D process grid 1 × gcds-per-node × nnodes when @p num_procs is a
 *        multiple of `kSpectralNodeGcds` and both axes divide @p size.
 *
 * Rank order is x-fastest, so ranks `[node*8, node*8+7]` share a z-slab of
 * the node and map onto Slurm's 8-ranks-per-node layout. Returns `{0,0,0}`
 * when this layout does not divide the grid.
 */
[[nodiscard]] inline Int3 node_aware_fft_proc_grid(const Int3 &size, int num_procs) {
  if (num_procs < kSpectralSlabMinRanks || num_procs % kSpectralNodeGcds != 0) {
    return Int3{0, 0, 0};
  }
  const int nnodes = num_procs / kSpectralNodeGcds;
  if (size[1] % kSpectralNodeGcds == 0 && size[2] % nnodes == 0) {
    return Int3{1, kSpectralNodeGcds, nnodes};
  }
  if (size[2] % kSpectralNodeGcds == 0 && size[1] % nnodes == 0) {
    return Int3{1, nnodes, kSpectralNodeGcds};
  }
  return Int3{0, 0, 0};
}

/// `OPENPFC_FFT_NODE_GRID=1` selects @ref node_aware_fft_proc_grid (1×8×N).
[[nodiscard]] inline bool fft_node_grid_override() {
  const char *e = std::getenv("OPENPFC_FFT_NODE_GRID");
  return e != nullptr && e[0] == '1' && e[1] == '\0';
}

/// `OPENPFC_FFT_PROC_GRID=gx,gy,gz` (or `gx x gy x gz`). `{0,0,0}` if unset.
[[nodiscard]] inline Int3 fft_proc_grid_override() {
  const char *e = std::getenv("OPENPFC_FFT_PROC_GRID");
  if (e == nullptr || e[0] == '\0') {
    return Int3{0, 0, 0};
  }
  int g[3] = {0, 0, 0};
  const char *p = e;
  for (int i = 0; i < 3; ++i) {
    char *end = nullptr;
    const long v = std::strtol(p, &end, 10);
    if (end == p || v < 1 || v > 1024) {
      return Int3{0, 0, 0};
    }
    g[i] = static_cast<int>(v);
    if (i < 2) {
      if (*end != ',' && *end != 'x' && *end != 'X') {
        return Int3{0, 0, 0};
      }
      p = end + 1;
    } else if (*end != '\0') {
      return Int3{0, 0, 0};
    }
  }
  return Int3{g[0], g[1], g[2]};
}

/// Brick min-surface on one node; 1D slabs off-node (measured fastest on
/// LUMI-G). `OPENPFC_FFT_NODE_GRID=1` selects the 1×8×N pencil grid.
/// `OPENPFC_FFT_PROC_GRID=gx,gy,gz` forces that Cartesian grid when it
/// factors `num_procs` and divides `size`.
[[nodiscard]] inline Int3 spectral_fft_proc_grid(const Int3 &size, int num_procs,
                                                 int r2c_direction = 0) {
  const Int3 forced = fft_proc_grid_override();
  if (forced[0] * forced[1] * forced[2] == num_procs && forced[0] >= 1 &&
      size[0] % forced[0] == 0 && size[1] % forced[1] == 0 &&
      size[2] % forced[2] == 0) {
    return forced;
  }
  if (num_procs >= kSpectralSlabMinRanks) {
    if (fft_node_grid_override()) {
      const Int3 node = node_aware_fft_proc_grid(size, num_procs);
      if (node[0] * node[1] * node[2] == num_procs) {
        return node;
      }
    }
    return slab_proc_grid(size, num_procs, r2c_direction);
  }
  return min_surface_proc_grid(size, num_procs);
}

/// Regular Cartesian split of an inclusive box; ranks are x-fastest.
[[nodiscard]] inline std::vector<Box3i> split_box(const Box3i &world,
                                                  const Int3 &grid) {
  if (!world.is_consistent()) {
    throw std::invalid_argument("split_box: world box is not consistent");
  }
  for (int d = 0; d < 3; ++d) {
    if (grid[d] < 1) {
      throw std::invalid_argument("split_box: process grid must be >= 1");
    }
    if (grid[d] > world.size[d]) {
      throw std::invalid_argument("split_box: more ranks than cells on an axis");
    }
  }

  auto cut = [&](int axis, int i) -> int {
    const int n = world.size[axis];
    const int g = grid[axis];
    return world.low[axis] + i * (n / g) + std::min(i, n % g);
  };

  std::vector<Box3i> out;
  out.reserve(static_cast<std::size_t>(grid[0]) * static_cast<std::size_t>(grid[1]) *
              static_cast<std::size_t>(grid[2]));
  for (int k = 0; k < grid[2]; ++k) {
    for (int j = 0; j < grid[1]; ++j) {
      for (int i = 0; i < grid[0]; ++i) {
        const std::array<int, 3> lo{cut(0, i), cut(1, j), cut(2, k)};
        const std::array<int, 3> hi{cut(0, i + 1) - 1, cut(1, j + 1) - 1,
                                    cut(2, k + 1) - 1};
        out.push_back(Box3i::from_bounds(lo, hi));
      }
    }
  }
  return out;
}

} // namespace pfc::decomposition
