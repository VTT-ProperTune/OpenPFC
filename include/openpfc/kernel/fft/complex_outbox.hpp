// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file complex_outbox.hpp
 * @brief r2c complex-outbox process-grid selection via HeFFTe plan costs
 *
 * @details
 * HeFFTe `split_world()` assigns approximately equal extents; exact
 * divisibility is not required. The production selector enumerates legal
 * complex process grids and minimises distributed reshape cost using the
 * same `heffte::plan_options` that will construct the FFT instance.
 *
 * Ranking:
 * 1. fewer MPI reshapes (`plan_operations` + `heffte::match`);
 * 2. fewer predicted all-peer rounds (2 per MPI reshape per cycle);
 * 3. lower local-box volume imbalance;
 * 4. lower local-box aspect ratio;
 * 5. lexicographic process-grid tie break.
 *
 * `plan_operations` reads `use_pencils`, `use_reorder`, and subranks.
 * `reshape_algorithm` and `use_gpu_aware` affect reshape construction, not
 * the logic plan; tests pin that those two fields do not change the
 * selected grid or MPI-reshape count.
 *
 * Overrides, fail-closed:
 *
 * - `OPENPFC_FFT_COMPLEX_PROC_GRID=gx,gy,gz` (or `gx x gy x gz`)
 * - `OPENPFC_FFT_COMPLEX_OUTBOX=legacy` restores the even-divisibility
 *   y-slab heuristic; unset or `min_reshape` uses the selector
 */

#ifndef OPENPFC_KERNEL_FFT_COMPLEX_OUTBOX_HPP
#define OPENPFC_KERNEL_FFT_COMPLEX_OUTBOX_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <heffte.h>

#include <openpfc/kernel/data/types.hpp>

namespace pfc::fft::layout {

using pfc::types::Int3;

inline constexpr const char *kComplexProcGridEnv = "OPENPFC_FFT_COMPLEX_PROC_GRID";
inline constexpr const char *kComplexOutboxEnv = "OPENPFC_FFT_COMPLEX_OUTBOX";

enum class ReshapeKind { Identity, Local, Mpi };
enum class ComplexOutboxMode { MinReshape, Legacy };

struct ComplexOutboxCost {
  Int3 real_grid{};
  Int3 complex_grid{};
  Int3 local_box_min{};
  Int3 local_box_max{};
  int n_reshape_stages = 0;
  int n_mpi_reshapes = 0;
  int n_local_reshapes = 0;
  int n_identity = 0;
  int predicted_all_peer_rounds_per_cycle = 0;
  double volume_imbalance = 1.0;
  double box_aspect = 1.0;
  long long min_box_count = 0;
  long long max_box_count = 0;
};

/// HeFFTe's r2c constructor applies `set_options<Backend, true>`.
template <typename BackendTag>
[[nodiscard]] inline heffte::plan_options
effective_r2c_plan_options(heffte::plan_options options) {
  return heffte::set_options<BackendTag, true>(options);
}

[[nodiscard]] inline heffte::box3d<int> real_world_box(const Int3 &size) {
  return heffte::box3d<int>({0, 0, 0}, {size[0] - 1, size[1] - 1, size[2] - 1});
}

[[nodiscard]] inline heffte::box3d<int> r2c_complex_world_box(const Int3 &size,
                                                              int r2c_direction) {
  if (r2c_direction == 0) {
    return heffte::box3d<int>({0, 0, 0}, {size[0] / 2, size[1] - 1, size[2] - 1});
  }
  if (r2c_direction == 1) {
    return heffte::box3d<int>({0, 0, 0}, {size[0] - 1, size[1] / 2, size[2] - 1});
  }
  if (r2c_direction == 2) {
    return heffte::box3d<int>({0, 0, 0}, {size[0] - 1, size[1] - 1, size[2] / 2});
  }
  throw std::invalid_argument("r2c_complex_world_box: r2c_direction must be 0, 1, "
                              "or 2, got " +
                              std::to_string(r2c_direction));
}

[[nodiscard]] inline int proc_grid_nproc(const Int3 &grid) {
  return grid[0] * grid[1] * grid[2];
}

[[nodiscard]] inline std::string format_proc_grid(const Int3 &grid) {
  return std::to_string(grid[0]) + "x" + std::to_string(grid[1]) + "x" +
         std::to_string(grid[2]);
}

[[nodiscard]] inline bool
split_boxes_are_legal(const std::vector<heffte::box3d<int>> &boxes,
                      const heffte::box3d<int> &world) {
  if (boxes.empty()) {
    return false;
  }
  for (const auto &b : boxes) {
    if (b.empty()) {
      return false;
    }
  }
  try {
    return heffte::world_complete(boxes, world);
  } catch (const std::invalid_argument &) {
    return false;
  }
}

[[nodiscard]] inline bool
complex_proc_grid_is_legal(const heffte::box3d<int> &complex_world, const Int3 &grid,
                           int nproc) {
  if (nproc < 1 || proc_grid_nproc(grid) != nproc) {
    return false;
  }
  for (int d = 0; d < 3; ++d) {
    if (grid[d] < 1 || grid[d] > complex_world.size[d]) {
      return false;
    }
  }
  const auto boxes = heffte::split_world(complex_world, grid);
  return split_boxes_are_legal(boxes, complex_world);
}

/// Cartesian factorizations of `nproc` that fit in `world` (uneven OK).
[[nodiscard]] inline std::vector<Int3>
legal_complex_proc_grids(const heffte::box3d<int> &world, int nproc) {
  std::vector<Int3> out;
  if (nproc < 1) {
    return out;
  }
  const int nx = world.size[0];
  const int ny = world.size[1];
  const int nz = world.size[2];
  for (int i = 1; i <= nproc && i <= nx; ++i) {
    if (nproc % i != 0) {
      continue;
    }
    const int rest = nproc / i;
    for (int j = 1; j <= rest && j <= ny; ++j) {
      if (rest % j != 0) {
        continue;
      }
      const int k = rest / j;
      if (k < 1 || k > nz) {
        continue;
      }
      const Int3 g{i, j, k};
      if (complex_proc_grid_is_legal(world, g, nproc)) {
        out.push_back(g);
      }
    }
  }
  return out;
}

[[nodiscard]] inline ReshapeKind
classify_reshape(const std::vector<heffte::box3d<int>> &in_boxes,
                 const std::vector<heffte::box3d<int>> &out_boxes) {
  if (heffte::match(in_boxes, out_boxes)) {
    if (in_boxes.empty() || in_boxes[0].ordered_same_as(out_boxes[0])) {
      return ReshapeKind::Identity;
    }
    return ReshapeKind::Local;
  }
  return ReshapeKind::Mpi;
}

[[nodiscard]] inline ComplexOutboxCost
plan_r2c_outbox_cost(const heffte::box3d<int> &real_world, const Int3 &real_grid,
                     const heffte::box3d<int> &complex_world,
                     const Int3 &complex_grid, int r2c_direction,
                     const heffte::plan_options &options) {
  const int nproc = proc_grid_nproc(real_grid);
  if (!complex_proc_grid_is_legal(complex_world, complex_grid, nproc)) {
    throw std::invalid_argument(
        "plan_r2c_outbox_cost: illegal complex process grid " +
        format_proc_grid(complex_grid));
  }
  heffte::ioboxes<int> boxes;
  boxes.in = heffte::split_world(real_world, real_grid);
  boxes.out = heffte::split_world(complex_world, complex_grid);
  if (!split_boxes_are_legal(boxes.in, real_world)) {
    throw std::invalid_argument("plan_r2c_outbox_cost: illegal real process grid");
  }
  const auto plan = [&]() {
    try {
      return heffte::plan_operations(boxes, r2c_direction, options,
                                     /*mpi_rank=*/0);
    } catch (const std::runtime_error &e) {
      throw std::invalid_argument(
          std::string("plan_r2c_outbox_cost: HeFFTe cannot plan this pair: ") +
          e.what());
    }
  }();

  ComplexOutboxCost cost;
  cost.real_grid = real_grid;
  cost.complex_grid = complex_grid;
  cost.min_box_count = std::numeric_limits<long long>::max();
  cost.local_box_min = {std::numeric_limits<int>::max(),
                        std::numeric_limits<int>::max(),
                        std::numeric_limits<int>::max()};
  for (const auto &b : boxes.out) {
    cost.min_box_count = std::min(cost.min_box_count, b.count());
    cost.max_box_count = std::max(cost.max_box_count, b.count());
    for (int d = 0; d < 3; ++d) {
      cost.local_box_min[d] = std::min(cost.local_box_min[d], b.size[d]);
      cost.local_box_max[d] = std::max(cost.local_box_max[d], b.size[d]);
    }
  }
  if (cost.min_box_count > 0) {
    cost.volume_imbalance = static_cast<double>(cost.max_box_count) /
                            static_cast<double>(cost.min_box_count);
  }
  const int a = std::max(
      {cost.local_box_max[0], cost.local_box_max[1], cost.local_box_max[2]});
  const int bmin = std::min(
      {cost.local_box_min[0], cost.local_box_min[1], cost.local_box_min[2]});
  if (bmin > 0) {
    cost.box_aspect = static_cast<double>(a) / static_cast<double>(bmin);
  }
  for (int i = 0; i < 4; ++i) {
    const ReshapeKind kind = classify_reshape(plan.in_shape[i], plan.out_shape[i]);
    if (kind == ReshapeKind::Identity) {
      ++cost.n_identity;
    } else if (kind == ReshapeKind::Local) {
      ++cost.n_local_reshapes;
      ++cost.n_reshape_stages;
    } else {
      ++cost.n_mpi_reshapes;
      ++cost.n_reshape_stages;
    }
  }
  cost.predicted_all_peer_rounds_per_cycle = 2 * cost.n_mpi_reshapes;
  return cost;
}

[[nodiscard]] inline bool complex_outbox_cost_better(const ComplexOutboxCost &a,
                                                     const ComplexOutboxCost &b) {
  if (a.n_mpi_reshapes != b.n_mpi_reshapes) {
    return a.n_mpi_reshapes < b.n_mpi_reshapes;
  }
  if (a.predicted_all_peer_rounds_per_cycle !=
      b.predicted_all_peer_rounds_per_cycle) {
    return a.predicted_all_peer_rounds_per_cycle <
           b.predicted_all_peer_rounds_per_cycle;
  }
  if (std::abs(a.volume_imbalance - b.volume_imbalance) > 1.0e-12) {
    return a.volume_imbalance < b.volume_imbalance;
  }
  if (std::abs(a.box_aspect - b.box_aspect) > 1.0e-12) {
    return a.box_aspect < b.box_aspect;
  }
  return a.complex_grid < b.complex_grid;
}

[[nodiscard]] inline Int3 select_min_distributed_reshape_complex_grid(
    const heffte::box3d<int> &real_world, const Int3 &real_grid,
    const heffte::box3d<int> &complex_world, int r2c_direction,
    const heffte::plan_options &options) {
  const int nproc = proc_grid_nproc(real_grid);
  const auto grids = legal_complex_proc_grids(complex_world, nproc);
  if (grids.empty()) {
    throw std::invalid_argument(
        "select_min_distributed_reshape_complex_grid: no legal complex "
        "process grid");
  }
  bool have = false;
  ComplexOutboxCost best{};
  Int3 winner{};
  for (const Int3 &g : grids) {
    try {
      const ComplexOutboxCost c = plan_r2c_outbox_cost(
          real_world, real_grid, complex_world, g, r2c_direction, options);
      if (!have || complex_outbox_cost_better(c, best)) {
        best = c;
        winner = g;
        have = true;
      }
    } catch (const std::invalid_argument &) {
      continue;
    }
  }
  if (!have) {
    throw std::invalid_argument(
        "select_min_distributed_reshape_complex_grid: no HeFFTe-plannable "
        "complex process grid");
  }
  return winner;
}

[[nodiscard]] inline Int3 parse_complex_proc_grid_override() {
  const char *e = std::getenv(kComplexProcGridEnv);
  if (e == nullptr || e[0] == '\0') {
    return Int3{0, 0, 0};
  }
  int g[3] = {0, 0, 0};
  const char *p = e;
  for (int i = 0; i < 3; ++i) {
    char *end = nullptr;
    const long v = std::strtol(p, &end, 10);
    if (end == p || v < 1 || v > 1024) {
      throw std::invalid_argument(
          std::string(kComplexProcGridEnv) +
          " must be gx,gy,gz with each factor in [1, 1024], got \"" + e + "\"");
    }
    g[i] = static_cast<int>(v);
    if (i < 2) {
      if (*end != ',' && *end != 'x' && *end != 'X') {
        throw std::invalid_argument(std::string(kComplexProcGridEnv) +
                                    " must be gx,gy,gz or gx x gy x gz, got \"" + e +
                                    "\"");
      }
      p = end + 1;
    } else if (*end != '\0') {
      throw std::invalid_argument(std::string(kComplexProcGridEnv) +
                                  " has trailing characters: \"" + e + "\"");
    }
  }
  return Int3{g[0], g[1], g[2]};
}

[[nodiscard]] inline ComplexOutboxMode complex_outbox_mode() {
  const char *e = std::getenv(kComplexOutboxEnv);
  if (e == nullptr || e[0] == '\0' || std::string(e) == "min_reshape") {
    return ComplexOutboxMode::MinReshape;
  }
  if (std::string(e) == "legacy") {
    return ComplexOutboxMode::Legacy;
  }
  throw std::invalid_argument(
      std::string(kComplexOutboxEnv) +
      " must be unset, \"min_reshape\", or \"legacy\", got \"" + e + "\"");
}

/// Even-divisibility y-slab when r2c is x and the real inbox is a z-slab.
[[nodiscard]] inline Int3
legacy_complex_proc_grid_for_r2c(const Int3 &real_grid,
                                 const heffte::box3d<int> &complex_world,
                                 int r2c_direction) {
  if (r2c_direction == 0 && real_grid[0] == 1 && real_grid[1] == 1 &&
      real_grid[2] > 1) {
    const int n = real_grid[2];
    const int ny = complex_world.size[1];
    if (n > 0 && ny % n == 0) {
      return Int3{1, n, 1};
    }
  }
  return real_grid;
}

[[nodiscard]] inline Int3
complex_proc_grid_for_r2c(const Int3 &real_grid,
                          const heffte::box3d<int> &real_world,
                          const heffte::box3d<int> &complex_world, int r2c_direction,
                          const heffte::plan_options &options) {
  const int nproc = proc_grid_nproc(real_grid);
  const Int3 forced = parse_complex_proc_grid_override();
  if (forced[0] > 0) {
    if (!complex_proc_grid_is_legal(complex_world, forced, nproc)) {
      throw std::invalid_argument(std::string(kComplexProcGridEnv) + "=" +
                                  format_proc_grid(forced) +
                                  " is not a legal complex process grid for "
                                  "this r2c world");
    }
    return forced;
  }
  if (complex_outbox_mode() == ComplexOutboxMode::Legacy) {
    return legacy_complex_proc_grid_for_r2c(real_grid, complex_world, r2c_direction);
  }
  return select_min_distributed_reshape_complex_grid(
      real_world, real_grid, complex_world, r2c_direction, options);
}

} // namespace pfc::fft::layout

#endif
