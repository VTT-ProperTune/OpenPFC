// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file mpi_timer.hpp
 * @brief Tiny `tic` / `toc` wall-clock helper for MPI parallel sections.
 *
 * @details
 * Captures the canonical "barrier, time, time, allreduce-max" pattern that
 * every MPI app rewrites by hand to report timing of a parallel section:
 *
 *     pfc::runtime::MpiTimer timer{MPI_COMM_WORLD};
 *     pfc::runtime::tic(timer);
 *     // ... time-stepping loop or other parallel section ...
 *     const double max_elapsed = pfc::runtime::toc(timer);
 *
 * `tic` issues an `MPI_Barrier` so all ranks agree on the start instant,
 * then records the local `MPI_Wtime()`. `toc` reads `MPI_Wtime()` again on
 * the calling rank and `MPI_Allreduce`s the local elapsed with `MPI_MAX`,
 * so every rank gets the **slowest rank's wall-clock** — the conventional
 * "wall time of the parallel section" reported by HPC apps.
 *
 * @note `MPI_MAX` is the only reduction provided here because it is what
 *       the existing OpenPFC apps actually want. Callers that need a
 *       non-collective local elapsed (e.g. asymmetric per-rank setup
 *       phases) can compute it directly: `MPI_Wtime() - timer.t_start`.
 *
 * @see openpfc/runtime/common/cpu_affinity.hpp for the sibling host-side
 *      runtime helper that lives in this same `pfc::runtime` namespace.
 */

#include <mpi.h>

namespace pfc::runtime {

/**
 * @brief Lightweight handle that pairs an MPI communicator with the start
 *        instant captured by the most recent `tic(timer)` call.
 *
 * Default-constructed timers use `MPI_COMM_WORLD`; pass a different
 * communicator at construction to time sub-communicators.
 */
struct MpiTimer {
  MPI_Comm comm{MPI_COMM_WORLD};
  double t_start{0.0};
};

/**
 * @brief Synchronize ranks on `timer.comm` and start the clock.
 *
 * Calls `MPI_Barrier` so every rank agrees on the start instant, then
 * records `MPI_Wtime()` into `timer.t_start`. Pair with `toc`.
 */
inline void tic(MpiTimer &timer) noexcept {
  MPI_Barrier(timer.comm);
  timer.t_start = MPI_Wtime();
}

/**
 * @brief Stop the clock and return the wall time of the **slowest rank**.
 *
 * Reads `MPI_Wtime()` on every rank, computes `local = now - t_start`, and
 * `MPI_Allreduce`s with `MPI_MAX` over `timer.comm`. The same value is
 * returned on every rank — typically the caller prints it from rank 0.
 *
 * @note `toc` does **not** issue a trailing `MPI_Barrier`. The
 *       `MPI_Allreduce` is itself collective, so all ranks have already
 *       reached this point by the time it returns. Skipping the explicit
 *       barrier saves one round-trip when the caller only needs the
 *       reduced max (the common case).
 */
inline double toc(const MpiTimer &timer) noexcept {
  const double local = MPI_Wtime() - timer.t_start;
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, timer.comm);
  return global;
}

} // namespace pfc::runtime
