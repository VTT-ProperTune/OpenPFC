// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file mpi_main.hpp
 * @brief Generic MPI app entry-point wrapper.
 *
 * @details
 * Every OpenPFC application's `main` opens with the same boilerplate:
 * `MPI_Init`, query `rank` and `nproc`, run the launcher's CPU-affinity
 * rescue, dispatch to the body, swallow any `std::exception` thrown by
 * the body via `MPI_Abort`, then `MPI_Finalize`. `pfc::runtime::mpi_main`
 * encapsulates that pattern so app `main`s reduce to one statement:
 *
 *     int main(int argc, char** argv) {
 *       return pfc::runtime::mpi_main(
 *           argc, argv, [](int argc, char** argv, int rank, int nproc) {
 *             // ... user body, return EXIT_SUCCESS / EXIT_FAILURE ...
 *           });
 *     }
 *
 * The body is invoked with the (possibly MPI-stripped) `argc` / `argv`
 * pair returned by `MPI_Init`, plus the rank and total rank count on
 * `MPI_COMM_WORLD`. The body's return value becomes `mpi_main`'s return
 * value (so a CLI parse failure can `return EXIT_FAILURE` without
 * needing an `MPI_Abort`).
 *
 * Exception handling matches the historical heat3d / tungsten pattern:
 * any `std::exception` from the body is logged on the offending rank's
 * `stderr` and the program is terminated with `MPI_Abort(comm, 1)`,
 * which prevents other ranks from hanging in subsequent collectives.
 *
 * @see openpfc/runtime/common/cpu_affinity.hpp for the CPU-affinity
 *      rescue this wrapper invokes.
 */

#include <cstdlib>
#include <exception>
#include <iostream>
#include <mpi.h>
#include <utility>

#include <openpfc/runtime/common/cpu_affinity.hpp>

namespace pfc::runtime {

/**
 * @brief MPI app `main` wrapper. Calls `body(argc, argv, rank, nproc)`.
 *
 * @tparam Body Callable invocable as `int(int, char**, int, int)`. The
 *              return value is propagated as `mpi_main`'s exit code on
 *              normal completion.
 *
 * @return The body's return value, or never returns when an exception
 *         escapes the body (control transfers to `MPI_Abort` and the
 *         program terminates).
 *
 * @note `MPI_Init` is allowed to consume MPI-specific arguments from
 *       `argc` / `argv`; the (possibly stripped) values are forwarded to
 *       the body.
 *
 * @note `mpi_main` does **not** rethrow caught exceptions: by the time
 *       the catch is reached we have already called `MPI_Abort`, which
 *       terminates the process. The catch only exists so we can log a
 *       useful message before the abort.
 */
template <class Body> int mpi_main(int argc, char **argv, Body &&body) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  reset_cpu_affinity_if_single_mpi_rank(nproc);

  int rc = EXIT_SUCCESS;
  try {
    rc = std::forward<Body>(body)(argc, argv, rank, nproc);
  } catch (const std::exception &e) {
    std::cerr << "(rank " << rank << "): " << e.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return rc;
}

} // namespace pfc::runtime
