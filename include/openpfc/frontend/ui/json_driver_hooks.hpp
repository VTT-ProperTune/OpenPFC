// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file json_driver_hooks.hpp
 * @brief One-call setup for JSON driver diagnostics (log rank + NaN checks)
 *
 * @details
 * JSON session drivers that parse FFT / HeFFTe options should align
 * `from_json` log prefixes with the MPI rank and ensure NaN-check macros use the
 * same communicator as the application. Call `configure_json_driver_hooks`
 * once the rank and `MPI_Comm` are known.
 */

#ifndef PFC_UI_JSON_DRIVER_HOOKS_HPP
#define PFC_UI_JSON_DRIVER_HOOKS_HPP

#include <mpi.h>
#include <string_view>

#include <openpfc/frontend/ui/from_json_log.hpp>
#include <openpfc/frontend/utils/nancheck.hpp>

namespace pfc::ui {

/** @brief Prefix for structured application log lines */
inline constexpr std::string_view k_app_log_tag = "[app] ";

/**
 * @brief Align `from_json` logging and default NaN-check communicator with the
 * driver
 *
 * @param comm Application communicator (same as `App` / session stack).
 * @param mpi_log_rank Rank index for `from_json` diagnostics (typically
 *        `MPI_Worker::get_rank()`).
 */
inline void configure_json_driver_hooks(MPI_Comm comm, int mpi_log_rank) {
  set_from_json_log_rank(mpi_log_rank);
  pfc::utils::set_default_nan_check_mpi_comm(comm);
}

} // namespace pfc::ui

#endif // PFC_UI_JSON_DRIVER_HOOKS_HPP
