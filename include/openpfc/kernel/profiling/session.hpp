// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file session.hpp
 * @brief Profiling frames (optional per-frame scalars), nested timed scopes, MPI
 * export (JSON and/or HDF5)
 */

#ifndef PFC_KERNEL_PROFILING_SESSION_HPP
#define PFC_KERNEL_PROFILING_SESSION_HPP

#include <openpfc/kernel/profiling/metric_catalog.hpp>

#include <nlohmann/json.hpp>

#include <iosfwd>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mpi.h>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace pfc::profiling {

namespace detail {
/// Unpack one row from the packed layout used by `MPI_Gatherv` / JSON export.
inline void unpack_gathered_profiling_row(std::size_t row, int stride, int nmeta,
                                          int kpaths,
                                          const std::vector<double> &flat,
                                          std::vector<double> &metrics,
                                          std::vector<double> &inc,
                                          std::vector<double> &exc) {
  const std::size_t b = row * static_cast<std::size_t>(stride);
  metrics.resize(static_cast<std::size_t>(nmeta));
  for (int m = 0; m < nmeta; ++m) {
    metrics[static_cast<std::size_t>(m)] = flat[b + static_cast<std::size_t>(m)];
  }
  inc.resize(static_cast<std::size_t>(kpaths));
  exc.resize(static_cast<std::size_t>(kpaths));
  for (int i = 0; i < kpaths; ++i) {
    const std::size_t base =
        b + static_cast<std::size_t>(nmeta) + (2U * static_cast<std::size_t>(i));
    inc[static_cast<std::size_t>(i)] = flat[base];
    exc[static_cast<std::size_t>(i)] = flat[base + 1U];
  }
}
} // namespace detail

struct ProfilingPrintOptions;

struct ProfilingExportOptions {
  bool write_json = true;
  bool write_hdf5 = false;
  std::string json_path;
  std::string hdf5_path;
  /// Empty: legacy layout (JSON/HDF5 schema v2 at `openpfc/profiling/` root).
  /// Non-empty: schema v3 with payload under `openpfc/profiling/runs/<sanitized>/`.
  std::string run_id;
  /// Optional key/value metadata (strings, numbers, bools, or nested JSON dumped
  /// as strings) written to HDF5 run-group attributes and JSON `metadata`.
  nlohmann::json export_metadata = nlohmann::json::object();
};

/// Sanitize @p run_id for use as a single HDF5 group name under `runs/`.
std::string sanitize_profiling_run_id_for_hdf5(std::string_view run_id);

/**
 * @brief Column-oriented frames; region catalog defines dense per-path
 * inclusive/exclusive times. Per-frame scalar metrics are configurable by name.
 */
class ProfilingSession {
public:
  /**
   * @brief Default frame metric names for the OpenPFC `App` (see
   *        `openpfc_frame_metrics.hpp`). Same as
   * `openpfc_default_frame_metric_names()`.
   */
  static std::vector<std::string> openpfc_default_frame_metrics();

  /**
   * @param catalog Region paths for inclusive/exclusive timers (tree export).
   * @param frame_metric_names Ordered names for one double per frame, e.g. latency
   *        or custom counters. Empty = no per-frame scalars (only region timers).
   */
  ProfilingSession(ProfilingMetricCatalog catalog,
                   std::vector<std::string> frame_metric_names = {});

  ProfilingSession(const ProfilingSession &) = delete;
  ProfilingSession &operator=(const ProfilingSession &) = delete;
  ProfilingSession(ProfilingSession &&) noexcept = default;
  ProfilingSession &operator=(ProfilingSession &&) noexcept = default;

  const ProfilingMetricCatalog &catalog() const noexcept { return catalog_; }

  const std::vector<std::string> &frame_metric_names() const noexcept {
    return frame_metric_names_;
  }

  /// Ensure @p path (and parent prefixes) exist in the catalog; may resize stored
  /// frames. All MPI ranks must register the same paths in the same order for valid
  /// gather.
  void ensure_path(std::string_view path) noexcept;

  /// Start a frame: clears region scratch and per-frame metric scratch.
  void begin_frame() noexcept;

  /// Set a per-frame scalar (ignored if @p name is not in frame_metric_names).
  void set_frame_metric(std::string_view name, double value) noexcept;

  /// Set @p name to elapsed seconds since @ref begin_frame (steady clock).
  void set_frame_metric_elapsed_since_begin(std::string_view name) noexcept;

  /// Commit region timers and append this frame (scopes must be balanced).
  void end_frame() noexcept;

  /// Manual additive time (inclusive == exclusive); ignored if path not in catalog.
  void add_recorded_time(std::string_view path, double seconds) noexcept;

  /// Set inclusive and exclusive for @p path (overwrites scratch); ignored if not in
  /// catalog.
  void assign_recorded_time(std::string_view path, double seconds) noexcept;

  void push_timed_scope(std::string_view path) noexcept;
  void pop_timed_scope() noexcept;

  /// Anchor for optional "wall clock since …" line in print_profiling_timer
  /// (TimerOutputs-style).
  void reset_report_clock() noexcept;

  std::size_t num_frames() const noexcept;

  void finalize_and_export(MPI_Comm comm,
                           const ProfilingExportOptions &options) const;

  /// Gather packed frames on rank @c 0 (same layout as finalize). On non-root,
  /// `row_counts`, `all_flat`, and `row_offset` are cleared.
  void mpi_gather_packed_frames(MPI_Comm comm, std::vector<int> &row_counts,
                                std::vector<double> &all_flat,
                                std::vector<std::size_t> &row_offset) const;

  friend void print_profiling_timer(std::ostream &os,
                                    const ProfilingSession &session,
                                    const ProfilingPrintOptions &opts);
  friend void print_profiling_timer(std::ostream &os, MPI_Comm comm,
                                    const ProfilingSession &session,
                                    const ProfilingPrintOptions &opts);

private:
  struct ScopeFrame {
    std::size_t path_index;
    std::chrono::steady_clock::time_point t0;
    double children_inclusive_sum{0};
  };

  int stride_doubles() const noexcept {
    return static_cast<int>(frame_metric_names_.size() + (2U * catalog_.size()));
  }

  void pack_frames_flat(std::vector<double> &out) const;

  void migrate_to_catalog(ProfilingMetricCatalog &&new_cat) noexcept;

  ProfilingMetricCatalog catalog_;
  std::vector<std::string> frame_metric_names_;
  std::unordered_map<std::string, std::size_t> frame_metric_ix_;
  /// Row-major: frame * n_meta + metric
  std::vector<double> frame_metric_values_;
  std::vector<double> frame_metric_scratch_;

  /// Row-major: frame * K + path_index
  std::vector<double> timer_inclusive_;
  std::vector<double> timer_exclusive_;

  std::vector<double> frame_scratch_inc_;
  std::vector<double> frame_scratch_exc_;
  std::vector<ScopeFrame> scope_stack_;

  bool frame_open_{false};
  std::chrono::steady_clock::time_point frame_wall_t0_;
  bool report_clock_valid_{false};
  std::chrono::steady_clock::time_point report_clock_origin_;
};

} // namespace pfc::profiling

#endif // PFC_KERNEL_PROFILING_SESSION_HPP
