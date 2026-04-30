// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file cpu_affinity.hpp
 * @brief Host CPU-affinity rescue for single-rank MPI + OpenMP launches.
 *
 * @details
 * Open MPI (and some other launchers) pin a single rank to one logical CPU,
 * so OpenMP sees `omp_get_num_procs() == 1` and all threads share one core.
 * For **exactly one MPI rank** on Linux, `reset_cpu_affinity_if_single_mpi_rank`
 * resets the process affinity mask to all online CPUs so OpenMP can scale.
 *
 * Multi-rank jobs are unchanged (each rank keeps the launcher's mask).
 * Opt out for an individual run with the environment variable
 * `OPENPFC_NO_RESET_AFFINITY` set to any value.
 *
 * On non-Linux platforms the function is a no-op.
 */

#include <cstdlib>

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

namespace pfc::runtime {

/**
 * @brief Restore an all-CPUs affinity mask when running with a single MPI rank.
 *
 * @param nproc Total number of MPI ranks in the communicator (typically the
 *              result of `MPI_Comm_size(MPI_COMM_WORLD, ...)`). A no-op when
 *              `nproc != 1`.
 */
inline void reset_cpu_affinity_if_single_mpi_rank(int nproc) noexcept {
#if defined(__linux__)
  if (nproc != 1) {
    return;
  }
  if (std::getenv("OPENPFC_NO_RESET_AFFINITY") != nullptr) {
    return;
  }
  const long ncpus = sysconf(_SC_NPROCESSORS_ONLN);
  if (ncpus <= 1) {
    return;
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  for (long i = 0; i < ncpus; ++i) {
    CPU_SET(static_cast<int>(i), &set);
  }
  (void)sched_setaffinity(0, sizeof(set), &set);
#else
  (void)nproc;
#endif
}

} // namespace pfc::runtime
