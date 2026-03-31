// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file timer_report.hpp
 * @brief TimerOutputs-style formatted table from ProfilingSession aggregates
 */

#ifndef PFC_KERNEL_PROFILING_TIMER_REPORT_HPP
#define PFC_KERNEL_PROFILING_TIMER_REPORT_HPP

#include <openpfc/kernel/profiling/session.hpp>

#include <iosfwd>
#include <string>

namespace pfc::profiling {

/// Options for console profiling table (TimerOutputs-inspired).
struct ProfilingPrintOptions {
  std::string title{"OpenPFC profiling"};
  /// Use ASCII '-' instead of Unicode box lines.
  bool ascii_lines{false};
  /// Sort sibling sections by total inclusive time (descending); else lexicographic.
  bool sort_by_time{true};
  /// Add an "exclusive" column (sum over frames).
  bool show_exclusive_column{false};
  /// Frame metric name whose values are summed for the %tot column; empty =
  /// denominator 1.0.
  std::string wall_denominator_metric{"wall_step"};
  /// When true and MPI size > 1, all ranks participate in a gather; rank 0 prints a
  /// table that combines per-rank timer totals (see mpi_aggregate_stat).
  bool mpi_aggregate_stdout{false};
  /// How to combine per-rank inclusive/exclusive totals: "mean", "sum", "min",
  /// "max", or "median". Ignored when mpi_aggregate_stdout is false.
  std::string mpi_aggregate_stat{"mean"};
};

/**
 * @brief Print hierarchical section / ncalls / time / %tot / avg for one session.
 *
 * Aggregates over all committed frames on this object (typically one MPI rank’s
 * rows unless you merge elsewhere). Denominator for %tot is the sum of
 * opts.wall_denominator_metric (default **wall_step**), or 1.0 if unset/unknown.
 */
void print_profiling_timer(std::ostream &os, const ProfilingSession &session,
                           const ProfilingPrintOptions &opts = {});

/**
 * @brief MPI-aware print: optional cross-rank aggregation on rank 0.
 *
 * If mpi_aggregate_stdout is false or @p comm has size 1, only rank 0 prints the
 * local session (same as print_profiling_timer(os, session, opts)). If
 * mpi_aggregate_stdout is true and size > 1, all ranks must call this; rank 0
 * gathers and prints combined statistics.
 */
void print_profiling_timer(std::ostream &os, MPI_Comm comm,
                           const ProfilingSession &session,
                           const ProfilingPrintOptions &opts);

/**
 * @brief Print using the thread-local active session (see ProfilingContextScope).
 *        No-op if no session or num_frames()==0.
 */
void print_profiling_timer(std::ostream &os, const ProfilingPrintOptions &opts);

} // namespace pfc::profiling

#endif // PFC_KERNEL_PROFILING_TIMER_REPORT_HPP
