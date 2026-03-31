// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file openpfc_frame_metrics.hpp
 * @brief Default per-frame scalar names and helpers for the OpenPFC simulator UI
 *
 * @details
 * `ProfilingSession` itself is generic (`begin_frame`, `set_frame_metric`,
 * `end_frame`). Step indices, MPI rank, barrier wall time, and heap byte samples are
 * **OpenPFC** conventions; use the helpers below from application code. Other
 * projects can ignore this header entirely.
 */

#ifndef PFC_KERNEL_PROFILING_OPENPFC_FRAME_METRICS_HPP
#define PFC_KERNEL_PROFILING_OPENPFC_FRAME_METRICS_HPP

#include <openpfc/kernel/profiling/names.hpp>
#include <openpfc/kernel/profiling/session.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace pfc::profiling {

/// String keys for `ProfilingSession::openpfc_default_frame_metrics()` (order
/// matters).
namespace openpfc_metrics {

inline constexpr std::string_view step = "step";
inline constexpr std::string_view mpi_rank = "mpi_rank";
inline constexpr std::string_view wall_step = "wall_step";
inline constexpr std::string_view rss_bytes = "rss_bytes";
inline constexpr std::string_view model_heap_bytes = "model_heap_bytes";
/// Second allocator bucket (in OpenPFC: FFT workspace); generic name for export.
inline constexpr std::string_view heap_secondary_bytes = "heap_secondary_bytes";

} // namespace openpfc_metrics

/// Ordered list matching @ref openpfc_metrics (for `ProfilingSession` construction).
inline std::vector<std::string> openpfc_default_frame_metric_names() {
  using namespace openpfc_metrics;
  return {std::string(step),
          std::string(mpi_rank),
          std::string(wall_step),
          std::string(rss_bytes),
          std::string(model_heap_bytes),
          std::string(heap_secondary_bytes)};
}

/// `begin_frame` then set step and MPI rank if those names exist in the session.
inline void openpfc_begin_frame_with_step_and_rank(ProfilingSession &s,
                                                   int step_index,
                                                   int rank_id) noexcept {
  s.begin_frame();
  s.set_frame_metric(openpfc_metrics::step, static_cast<double>(step_index));
  s.set_frame_metric(openpfc_metrics::mpi_rank, static_cast<double>(rank_id));
}

/**
 * @brief Set wall + memory scalars and @ref ProfilingSession::end_frame.
 *
 * Typical use after `openpfc_begin_frame_with_step_and_rank` and optional
 * `assign_recorded_time` / scopes. @p wall_step_seconds is often the barrier
 * duration from `measure_barriered`.
 */
inline void
openpfc_end_frame_step_wall_and_memory(ProfilingSession &s, double wall_step_seconds,
                                       std::uint64_t rss, std::uint64_t model_heap,
                                       std::uint64_t heap_secondary) noexcept {
  s.set_frame_metric(openpfc_metrics::wall_step, wall_step_seconds);
  s.set_frame_metric(openpfc_metrics::rss_bytes, static_cast<double>(rss));
  s.set_frame_metric(openpfc_metrics::model_heap_bytes,
                     static_cast<double>(model_heap));
  s.set_frame_metric(openpfc_metrics::heap_secondary_bytes,
                     static_cast<double>(heap_secondary));
  s.end_frame();
}

/// Like @ref openpfc_end_frame_step_wall_and_memory but wall = elapsed since
/// `begin_frame`.
inline void openpfc_end_frame_memory_only_wall_from_clock(
    ProfilingSession &s, std::uint64_t rss, std::uint64_t model_heap,
    std::uint64_t heap_secondary) noexcept {
  s.set_frame_metric_elapsed_since_begin(openpfc_metrics::wall_step);
  s.set_frame_metric(openpfc_metrics::rss_bytes, static_cast<double>(rss));
  s.set_frame_metric(openpfc_metrics::model_heap_bytes,
                     static_cast<double>(model_heap));
  s.set_frame_metric(openpfc_metrics::heap_secondary_bytes,
                     static_cast<double>(heap_secondary));
  s.end_frame();
}

/**
 * @brief Test/demo helper: set **fft** region time from a meter, then wall + memory,
 * then
 *        @ref ProfilingSession::end_frame.
 */
inline void openpfc_end_frame_with_fft_region_wall_and_memory(
    ProfilingSession &s, double wall_step_seconds, double fft_region_seconds,
    std::uint64_t rss, std::uint64_t model_heap,
    std::uint64_t heap_secondary) noexcept {
  s.assign_recorded_time(kProfilingRegionFft, fft_region_seconds);
  openpfc_end_frame_step_wall_and_memory(s, wall_step_seconds, rss, model_heap,
                                         heap_secondary);
}

} // namespace pfc::profiling

#endif // PFC_KERNEL_PROFILING_OPENPFC_FRAME_METRICS_HPP
