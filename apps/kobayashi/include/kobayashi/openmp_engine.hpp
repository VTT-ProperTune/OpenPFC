// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file openmp_engine.hpp
 * @brief Single-rank Kobayashi FD: `FDPaddedCPUStack` + OpenMP over owned cells.
 *
 * Periodic XY comes from `HaloExchange` (`Axes2D()`), same as
 * `kobayashi_fd_manual` at nproc=1. Initializes MPI if the caller has not.
 */

#include <cstddef>
#include <vector>

#include <kobayashi/cli.hpp>

namespace kobayashi::openmp_engine {

struct RunResult {
  std::vector<double> phi_xy;
  std::vector<double> tempr_xy;
  /** Wall seconds for the time-integration loop only. */
  double wall_loop_s = 0.0;
  /** Effective thread count (`omp_get_max_threads()` after optional override). */
  int nthreads = 1;
};

/**
 * Run the Kobayashi explicit FD integration on the full grid.
 *
 * If @p cfg.num_threads > 0, calls `omp_set_num_threads(cfg.num_threads)` before the
 * loop. Otherwise respects `OMP_NUM_THREADS` / runtime default.
 *
 * PNG snapshots match `kobayashi_fd_manual` when @p skip_png is false
 * (`phi_%04d.png`, final).
 */
RunResult run(const RunConfigOpenMP &cfg, bool skip_png, bool quiet);

} // namespace kobayashi::openmp_engine
