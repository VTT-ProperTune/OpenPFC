// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file from_json_log.hpp
 * @brief MPI rank prefix for `from_json` diagnostics
 */

#ifndef PFC_UI_FROM_JSON_LOG_HPP
#define PFC_UI_FROM_JSON_LOG_HPP

#include <atomic>

#include <openpfc/kernel/utils/logging.hpp>

namespace pfc::ui {

namespace detail {

inline std::atomic<int> &from_json_log_rank_storage() noexcept {
  static std::atomic<int> rank{-1};
  return rank;
}

} // namespace detail

/**
 * @brief MPI rank used when `from_json` helpers emit log lines (default: -1, no
 * prefix).
 *
 * Call `set_from_json_log_rank` from the application entry (e.g. `App::main` with
 * `MPI_Worker::get_rank()`) so FFT / plan-option messages are attributed to the
 * correct rank. Tests may set this for deterministic diagnostics.
 */
[[nodiscard]] inline int get_from_json_log_rank() noexcept {
  return detail::from_json_log_rank_storage().load(std::memory_order_relaxed);
}

inline void set_from_json_log_rank(int mpi_rank) noexcept {
  detail::from_json_log_rank_storage().store(mpi_rank, std::memory_order_relaxed);
}

/** @brief Logger for Info-level messages from `from_json` (rank from @ref
 * get_from_json_log_rank). */
[[nodiscard]] inline pfc::Logger from_json_info_logger() noexcept {
  return pfc::Logger{pfc::LogLevel::Info, get_from_json_log_rank()};
}

/** @brief Logger for Debug-level messages from `from_json` (rank from @ref
 * get_from_json_log_rank). */
[[nodiscard]] inline pfc::Logger from_json_debug_logger() noexcept {
  return pfc::Logger{pfc::LogLevel::Debug, get_from_json_log_rank()};
}

} // namespace pfc::ui

#endif // PFC_UI_FROM_JSON_LOG_HPP
