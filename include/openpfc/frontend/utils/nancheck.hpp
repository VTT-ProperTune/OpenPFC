// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file nancheck.hpp
 * @brief NaN detection and debugging utilities
 *
 * @details
 * This header provides macros and functions for detecting NaN (Not-a-Number)
 * values in floating-point computations, useful for debugging numerical issues.
 *
 * Key features:
 * - CHECK_AND_ABORT_IF_NAN(value): Macro to check single values; uses the
 *   **default NaN-check communicator** (see `default_nan_check_mpi_comm()`), which
 *   `pfc::ui::App::main` sets to the application communicator (otherwise
 *   `MPI_COMM_WORLD`).
 * - CHECK_AND_ABORT_IF_NANS(vec): Macro to check entire vectors (same default)
 * - CHECK_AND_ABORT_IF_NAN_MPI / CHECK_AND_ABORT_IF_NANS_MPI: pass an explicit
 *   communicator (e.g. `Model::mpi_comm()`) when the check site should not use
 *   the default
 * - Enabled by Debug builds or with OpenPFC_ENABLE_NAN_CHECK
 * - MPI-aware: Reports rank where NaN was detected
 *
 * NaN checks are enabled automatically in Debug builds. Enable them explicitly
 * in other build types with:
 * cmake -DOpenPFC_ENABLE_NAN_CHECK=ON
 *
 * @code
 * #include <openpfc/frontend/utils/nancheck.hpp>
 *
 * double result = compute_something();
 * CHECK_AND_ABORT_IF_NAN(result);  // Aborts if NaN detected
 * @endcode
 *
 * @see utils.hpp for other utility functions
 *
 * Use `CHECK_AND_ABORT_IF_NAN_MPI` / `CHECK_AND_ABORT_IF_NANS_MPI` when the
 * check must use a communicator other than the default (for example inside a
 * library that does not control `set_default_nan_check_mpi_comm`).
 */

#ifndef PFC_UTILS_NANCHECK_HPP
#define PFC_UTILS_NANCHECK_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <vector>

#if defined(NAN_CHECK_ENABLED)
/**
 * @def CHECK_AND_ABORT_IF_NAN(value)
 *
 * Macro for checking if a single float or double value is NaN and aborting
 * the application if NaN is detected.
 *
 * To enable NaN checks outside Debug builds, use the CMake option:
 *
 *     cmake -DOpenPFC_ENABLE_NAN_CHECK=ON path/to/source
 *
 * When enabled, the CHECK_AND_ABORT_IF_NAN macro will call the abortIfNaN
 * function to check the provided value for NaN. If NaN is detected, the
 * application will be aborted and an error message will be displayed,
 * indicating the process rank, file name, and line number where the NaN
 * was detected.
 *
 * Note: NaN checks may introduce a performance overhead and are typically
 * used for debugging and validation purposes. It is recommended to enable
 * NaN checks in debug builds and disable them in release builds to optimize
 * performance.
 */
#define CHECK_AND_ABORT_IF_NAN(value)                                               \
  ::pfc::utils::abortIfNaN((value), __FILE__, __LINE__,                             \
                           ::pfc::utils::default_nan_check_mpi_comm())

/**
 * @def CHECK_AND_ABORT_IF_NAN_MPI(value, comm)
 *
 * Same as @ref CHECK_AND_ABORT_IF_NAN but uses @p comm for rank and `MPI_Abort`.
 */
#define CHECK_AND_ABORT_IF_NAN_MPI(value, comm)                                     \
  ::pfc::utils::abortIfNaN((value), __FILE__, __LINE__, (comm))

/**
 * @def CHECK_AND_ABORT_IF_NANS(vec)
 *
 * Macro for checking if any NaNs are present in a vector of float or double
 * values and aborting the application if NaNs are detected.
 *
 * To enable NaN checks outside Debug builds, use the CMake option:
 *
 *     cmake -DOpenPFC_ENABLE_NAN_CHECK=ON path/to/source
 *
 * When enabled, the CHECK_AND_ABORT_IF_NANS macro will call the abortIfNaNs
 * function to check the vector for NaNs. If NaNs are detected, the
 * application will be aborted and an error message will be displayed,
 * indicating the process rank, file name, and line number where the NaNs
 * were detected.
 *
 * Note: NaN checks may introduce a performance overhead and are typically
 * used for debugging and validation purposes. It is recommended to enable
 * NaN checks in debug builds and disable them in release builds to optimize
 * performance.
 */
#define CHECK_AND_ABORT_IF_NANS(vec)                                                \
  ::pfc::utils::abortIfNaNs((vec), __FILE__, __LINE__,                              \
                            ::pfc::utils::default_nan_check_mpi_comm())

/**
 * @def CHECK_AND_ABORT_IF_NANS_MPI(vec, comm)
 *
 * Same as @ref CHECK_AND_ABORT_IF_NANS but uses @p comm for rank and `MPI_Abort`.
 */
#define CHECK_AND_ABORT_IF_NANS_MPI(vec, comm)                                      \
  ::pfc::utils::abortIfNaNs((vec), __FILE__, __LINE__, (comm))
#else
#define CHECK_AND_ABORT_IF_NAN(value)
#define CHECK_AND_ABORT_IF_NANS(vec)
#define CHECK_AND_ABORT_IF_NAN_MPI(value, comm)
#define CHECK_AND_ABORT_IF_NANS_MPI(vec, comm)
#endif

namespace pfc::utils {

namespace detail {
inline MPI_Comm &nan_check_default_comm_slot() {
  static MPI_Comm comm = MPI_COMM_WORLD;
  return comm;
}
} // namespace detail

/**
 * @brief Communicator used by @ref CHECK_AND_ABORT_IF_NAN and
 *        @ref CHECK_AND_ABORT_IF_NANS for rank reporting and `MPI_Abort`.
 *
 * Initially `MPI_COMM_WORLD`. `pfc::ui::App::main` calls
 * `set_default_nan_check_mpi_comm` with the application communicator so split
 * communicators behave correctly. Standalone drivers may set the default once
 * at startup.
 */
inline MPI_Comm default_nan_check_mpi_comm() {
  return detail::nan_check_default_comm_slot();
}

/**
 * @brief Install the communicator for the non-`_MPI` NaN check macros.
 *
 * @param comm Communicator for `MPI_Comm_rank` / `MPI_Abort`. If @p comm is
 *        `MPI_COMM_NULL`, the default falls back to `MPI_COMM_WORLD`.
 */
inline void set_default_nan_check_mpi_comm(MPI_Comm comm) {
  detail::nan_check_default_comm_slot() =
      (comm == MPI_COMM_NULL) ? MPI_COMM_WORLD : comm;
}

/**
 * Checks if there are any NaNs in a vector of floats.
 *
 * @param vec The vector of floats to check.
 * @return True if NaNs are found, false otherwise.
 */
template <typename T> bool hasNaNs(const std::vector<T> &vec) {
  return std::any_of(vec.begin(), vec.end(),
                     [](const T &value) { return std::isnan(value); });
}

template <typename T>
void abortIfNaN(T value, const char *filename, int line,
                MPI_Comm comm = MPI_COMM_WORLD) {
  if (std::isnan(value)) {
    int rank = 0;
    MPI_Comm c = (comm == MPI_COMM_NULL) ? MPI_COMM_WORLD : comm;
    MPI_Comm_rank(c, &rank);
    std::cerr << "NaN detected on process " << rank << " at " << filename << ":"
              << line << ". Aborting MPI application." << '\n';
    MPI_Abort(c, 1);
  }
}

/**
 * Checks if there are any NaNs in a vector of floats and aborts the MPI
 * application if NaNs are detected. Prints an error message indicating the
 * process rank where NaNs were found before aborting.
 *
 * @param vec The vector of floats to check.
 */
template <typename T>
void abortIfNaNs(const std::vector<T> &vec, const char *filename, int line,
                 MPI_Comm comm = MPI_COMM_WORLD) {
  if (hasNaNs(vec)) {
    int rank = 0;
    MPI_Comm c = (comm == MPI_COMM_NULL) ? MPI_COMM_WORLD : comm;
    MPI_Comm_rank(c, &rank);
    std::cerr << "NaNs detected on process " << rank << " at " << filename << ":"
              << line << ". Aborting MPI application." << '\n';
    MPI_Abort(c, 1);
  }
}

} // namespace pfc::utils

#endif // PFC_UTILS_NANCHECK_HPP
