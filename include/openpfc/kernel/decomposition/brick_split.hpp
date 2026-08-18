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
