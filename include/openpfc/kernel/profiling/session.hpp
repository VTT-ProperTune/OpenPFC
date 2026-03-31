// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file session.hpp
 * @brief Per-step profiling frames, nested timed scopes, MPI export (JSON / CSV /
 * HDF5)
 */

#ifndef PFC_KERNEL_PROFILING_SESSION_HPP
#define PFC_KERNEL_PROFILING_SESSION_HPP

#include <openpfc/kernel/profiling/metric_catalog.hpp>

#include <iosfwd>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mpi.h>
#include <string>
#include <string_view>
#include <vector>

namespace pfc::profiling {

struct ProfilingPrintOptions;

struct ProfilingExportOptions {
  bool write_json = true;
  bool write_csv = false;
  bool write_hdf5 = false;
  std::string json_path;
  std::string csv_path;
  std::string hdf5_path;
};

/**
 * @brief Column-oriented frames; catalog defines dense per-path inclusive/exclusive
 * times.
 */
class ProfilingSession {
public:
  explicit ProfilingSession(ProfilingMetricCatalog catalog);

  ProfilingSession(const ProfilingSession &) = delete;
  ProfilingSession &operator=(const ProfilingSession &) = delete;
  ProfilingSession(ProfilingSession &&) noexcept = default;
  ProfilingSession &operator=(ProfilingSession &&) noexcept = default;

  const ProfilingMetricCatalog &catalog() const noexcept { return catalog_; }

  /// Ensure @p path (and parent prefixes) exist in the catalog; may resize stored
  /// frames. All MPI ranks must register the same paths in the same order for valid
  /// gather.
  void ensure_path(std::string_view path) noexcept;

  void begin_step_frame(int step_index, int mpi_rank) noexcept;

  /// Manual additive time (inclusive == exclusive); ignored if path not in catalog.
  void add_recorded_time(std::string_view path, double seconds) noexcept;

  /// Set inclusive and exclusive for @p path (overwrites scratch); ignored if not in
  /// catalog.
  void assign_recorded_time(std::string_view path, double seconds) noexcept;

  void push_timed_scope(std::string_view path) noexcept;
  void pop_timed_scope() noexcept;

  /**
   * @brief Set per-frame wall_step metadata (e.g. MPI barrier duration). Call after
   *        begin_step_frame and before end_step_frame(rss, …). If omitted, wall_step
   * is steady_clock elapsed since begin_step_frame.
   */
  void set_frame_wall_step(double seconds) noexcept;

  /**
   * @brief Commit the frame: pop scopes, store memory columns, append row. Region
   * times come only from scopes, add_recorded_time, and assign_recorded_time.
   */
  void end_step_frame(std::uint64_t rss_bytes, std::uint64_t model_heap_bytes,
                      std::uint64_t fft_heap_bytes) noexcept;

  /// Convenience: fixed wall_step and fft region overwrite (tests / legacy callers).
  void end_step_frame(double wall_step_seconds, double fft_seconds,
                      std::uint64_t rss_bytes, std::uint64_t model_heap_bytes,
                      std::uint64_t fft_heap_bytes) noexcept;

  /// Anchor for optional "wall clock since …" line in print_profiling_timer
  /// (TimerOutputs-style).
  void reset_report_clock() noexcept;

  std::size_t num_frames() const noexcept { return step_index_.size(); }

  void finalize_and_export(MPI_Comm comm,
                           const ProfilingExportOptions &options) const;

  friend void print_profiling_timer(std::ostream &os,
                                    const ProfilingSession &session,
                                    const ProfilingPrintOptions &opts);

private:
  struct ScopeFrame {
    std::size_t path_index;
    std::chrono::steady_clock::time_point t0;
    double children_inclusive_sum{0};
  };

  static constexpr int kMetaCols = 6;
  int stride_doubles() const noexcept {
    return kMetaCols + static_cast<int>(2 * catalog_.size());
  }

  void pack_frames_flat(std::vector<double> &out) const;

  void commit_frame(double wall_step_seconds, bool inject_fft_region,
                    double fft_seconds, std::uint64_t rss_bytes,
                    std::uint64_t model_heap_bytes,
                    std::uint64_t fft_heap_bytes) noexcept;

  void migrate_to_catalog(ProfilingMetricCatalog &&new_cat) noexcept;

  ProfilingMetricCatalog catalog_;
  std::vector<int> step_index_;
  std::vector<int> mpi_rank_;
  std::vector<double> wall_step_;
  std::vector<double> rss_bytes_;
  std::vector<double> model_heap_bytes_;
  std::vector<double> fft_heap_bytes_;
  /// Row-major: frame * K + path_index
  std::vector<double> timer_inclusive_;
  std::vector<double> timer_exclusive_;

  std::vector<double> frame_scratch_inc_;
  std::vector<double> frame_scratch_exc_;
  std::vector<ScopeFrame> scope_stack_;

  int pending_step_ = -1;
  int pending_rank_ = -1;
  std::chrono::steady_clock::time_point frame_wall_t0_{};
  bool wall_override_set_{false};
  double wall_override_value_{0.0};
  bool report_clock_valid_{false};
  std::chrono::steady_clock::time_point report_clock_origin_{};
};

} // namespace pfc::profiling

#endif // PFC_KERNEL_PROFILING_SESSION_HPP
