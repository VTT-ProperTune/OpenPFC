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

namespace pfc {
namespace profiling {

/// Options for console profiling table (TimerOutputs-inspired).
struct ProfilingPrintOptions {
  std::string title{"OpenPFC profiling"};
  /// Use ASCII '-' instead of Unicode box lines.
  bool ascii_lines{false};
  /// Sort sibling sections by total inclusive time (descending); else lexicographic.
  bool sort_by_time{true};
  /// Add an "exclusive" column (sum over frames).
  bool show_exclusive_column{false};
};

/**
 * @brief Print hierarchical section / ncalls / time / %tot / avg for one session.
 *
 * Aggregates over all committed frames on this object (typically one MPI rank’s
 * rows unless you merge elsewhere). Denominator for %tot is sum of wall_step.
 */
void print_profiling_timer(std::ostream &os, const ProfilingSession &session,
                           const ProfilingPrintOptions &opts = {});

/**
 * @brief Print using the thread-local active session (see ProfilingContextScope).
 *        No-op if no session or num_frames()==0.
 */
void print_profiling_timer(std::ostream &os, const ProfilingPrintOptions &opts);

} // namespace profiling
} // namespace pfc

#endif // PFC_KERNEL_PROFILING_TIMER_REPORT_HPP
