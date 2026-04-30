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
 * ## Labeled sections (in-loop)
 *
 * For per-section breakdowns inside a hot loop ("inner", "halo_wait",
 * "border", "euler", ...) use the labeled overloads:
 *
 *     pfc::runtime::tic(timer, "inner");
 *     // ... compute inner stencil ...
 *     pfc::runtime::toc(timer, "inner");   // accumulates into the section
 *
 * Labeled `tic`/`toc` are **collective-free** (no `MPI_Barrier`,
 * no `MPI_Allreduce`) so they are safe to call thousands of times per
 * step without synchronisation overhead. Each label accumulates a
 * **local** wall-time total. Call `print_timing_summary(timer, rank)`
 * once at the end to allreduce-max each label and print a sorted table
 * on the chosen rank.
 *
 * @see openpfc/runtime/common/cpu_affinity.hpp for the sibling host-side
 *      runtime helper that lives in this same `pfc::runtime` namespace.
 */

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <map>
#include <mpi.h>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace pfc::runtime {

/**
 * @brief Lightweight handle that pairs an MPI communicator with the start
 *        instant captured by the most recent `tic(timer)` call.
 *
 * Default-constructed timers use `MPI_COMM_WORLD`; pass a different
 * communicator at construction to time sub-communicators.
 *
 * The `sections` map holds **local** accumulated wall-time per label
 * recorded by labeled `tic`/`toc` calls — see `print_timing_summary`
 * for the collective reduction that turns those local totals into a
 * global "slowest-rank" report.
 */
struct MpiTimer {
  MPI_Comm comm{MPI_COMM_WORLD};
  double t_start{0.0};
  /// Per-label state: `{ start, total_elapsed_local }` in seconds.
  std::map<std::string, std::pair<double, double>> sections{};

  MpiTimer() = default;
  explicit MpiTimer(MPI_Comm c) noexcept : comm(c) {}
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

/**
 * @brief Start a labeled section. **Collective-free** (no barrier).
 *
 * Records `MPI_Wtime()` as the start instant of `label` in
 * `timer.sections[label].first`. Multiple `tic(timer, label)` /
 * `toc(timer, label)` pairs accumulate into the same total, so a label
 * called once per step over `n_steps` iterations gives the total time
 * spent in that section.
 */
inline void tic(MpiTimer &timer, const std::string &label) {
  timer.sections[label].first = MPI_Wtime();
}

/**
 * @brief Stop a labeled section and accumulate the elapsed time locally.
 *
 * Returns the **local** elapsed wall-time of this `tic`/`toc` pair (no
 * reduction). The accumulated total is read by `print_timing_summary`.
 */
inline double toc(MpiTimer &timer, const std::string &label) {
  auto &slot = timer.sections[label];
  const double elapsed = MPI_Wtime() - slot.first;
  slot.second += elapsed;
  return elapsed;
}

/**
 * @brief Allreduce-max each labeled section and print a sorted table.
 *
 * For every label recorded by labeled `tic`/`toc` calls, computes the
 * `MPI_MAX` of the per-rank accumulated total over `timer.comm` and
 * prints a `label : seconds` line (largest total first) on the rank
 * matching `print_rank` (default rank 0). Returns the same vector of
 * `(label, max_total_seconds)` pairs on every rank, in print order, so
 * callers may emit additional formatting (e.g. CSV).
 *
 * @note Collective on `timer.comm` (one `MPI_Allreduce` per label).
 */
inline std::vector<std::pair<std::string, double>>
print_timing_summary(const MpiTimer &timer, int print_rank = 0,
                     std::ostream &os = std::cout) {
  std::vector<std::pair<std::string, double>> out;
  out.reserve(timer.sections.size());
  for (const auto &kv : timer.sections) {
    const double local = kv.second.second;
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, timer.comm);
    out.emplace_back(kv.first, global);
  }

  std::sort(out.begin(), out.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });

  int rank = 0;
  MPI_Comm_rank(timer.comm, &rank);
  if (rank == print_rank && !out.empty()) {
    std::size_t width = 0;
    for (const auto &kv : out) width = std::max(width, kv.first.size());
    os << "[timing] section breakdown (max over ranks):\n";
    for (const auto &kv : out) {
      os << "  " << std::left << std::setw(static_cast<int>(width)) << kv.first
         << " : " << std::fixed << std::setprecision(6) << kv.second << " s\n";
    }
  }
  return out;
}

} // namespace pfc::runtime
