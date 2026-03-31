// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file wall_clock.hpp
 * @brief MPI wall clock helpers and optional barrier-wrapped measurement
 *
 * Uses MPI_Wtime() for consistent semantics across MPI ranks. For code that
 * does not run under MPI, prefer std::chrono::steady_clock in the caller.
 *
 * Barriers synchronize ranks before/after the timed section: they measure
 * "everyone finished A, then everyone starts B" time, not overlapped
 * communication/compute. Document this when interpreting scalability results.
 */

#ifndef PFC_KERNEL_PROFILING_WALL_CLOCK_HPP
#define PFC_KERNEL_PROFILING_WALL_CLOCK_HPP

#include <mpi.h>
#include <utility>

namespace pfc {
namespace profiling {

/// Wall time in seconds from MPI (monotonic per rank).
inline double mpi_wtime_now() noexcept { return MPI_Wtime(); }

/**
 * @brief Run @p fn between MPI_Barrier calls and return elapsed seconds on
 *        this rank (time between barriers around @p fn).
 */
template <typename F> inline double measure_barriered(MPI_Comm comm, F &&fn) {
  MPI_Barrier(comm);
  const double t0 = MPI_Wtime();
  std::forward<F>(fn)();
  MPI_Barrier(comm);
  return MPI_Wtime() - t0;
}

/**
 * @brief Elapsed seconds from @p t0 to MPI_Wtime() (no barrier).
 */
inline double mpi_wtime_elapsed(double t0) noexcept { return MPI_Wtime() - t0; }

} // namespace profiling
} // namespace pfc

#endif // PFC_KERNEL_PROFILING_WALL_CLOCK_HPP
