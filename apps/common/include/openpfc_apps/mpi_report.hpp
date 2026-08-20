// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file mpi_report.hpp
 * @brief Rank-0 timing / scalar reductions for FD demo apps.
 */

#include <iostream>
#include <mpi.h>

namespace pfc::apps {

[[nodiscard]] inline double reduce_sum(double local, MPI_Comm comm, int root = 0) {
  double global = 0.0;
  MPI_Reduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, root, comm);
  return global;
}

[[nodiscard]] inline double reduce_max(double local, MPI_Comm comm, int root = 0) {
  double global = 0.0;
  MPI_Reduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, root, comm);
  return global;
}

inline void print_timing_line(std::ostream &os, double max_elapsed, int n_steps) {
  os << "timing_s=" << max_elapsed
     << " avg_step_time_s=" << (max_elapsed / static_cast<double>(n_steps))
     << " (MPI_MAX across ranks)\n";
}

/// Min/max/avg wall time of the time-step loop; rank 0 prints `Step timing:`.
inline void report_step_timing(MPI_Comm comm, int rank, int n_steps,
                               double elapsed_local_s) {
  double elapsed_min_s = 0.0;
  double elapsed_max_s = 0.0;
  double elapsed_sum_s = 0.0;
  MPI_Reduce(&elapsed_local_s, &elapsed_min_s, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
  MPI_Reduce(&elapsed_local_s, &elapsed_max_s, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&elapsed_local_s, &elapsed_sum_s, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

  int nproc = 1;
  MPI_Comm_size(comm, &nproc);
  if (rank == 0) {
    const double avg_elapsed_s = elapsed_sum_s / static_cast<double>(nproc);
    const double avg_step_time_s = elapsed_max_s / static_cast<double>(n_steps);
    std::cout << "Step timing: elapsed_min_s=" << elapsed_min_s
              << ", elapsed_max_s=" << elapsed_max_s
              << ", elapsed_avg_s=" << avg_elapsed_s
              << ", avg_step_time_s=" << avg_step_time_s << "\n";
  }
}

} // namespace pfc::apps
