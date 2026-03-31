// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file mpi_stats.hpp
 * @brief Reduce and gather per-rank scalars for timing and scalability analysis
 *
 * All ranks must participate in collective calls. Rank 0 receives full
 * per-rank arrays where noted.
 */

#ifndef PFC_KERNEL_PROFILING_MPI_STATS_HPP
#define PFC_KERNEL_PROFILING_MPI_STATS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <mpi.h>
#include <vector>

namespace pfc {
namespace profiling {

/**
 * @brief Statistics of one scalar across MPI ranks (valid on root after reduce).
 */
struct RankStats {
  double min = 0.0;
  double max = 0.0;
  double sum = 0.0;
  double mean = 0.0;
  /// Sample standard deviation (Bessel); 0 if count < 2.
  double stddev_sample = 0.0;
  int count = 0;
};

/**
 * @brief Gather each rank's value on root and compute min, max, sum, mean,
 *        sample standard deviation.
 *
 * Non-root ranks receive count set to comm size but min/max/mean/stddev are
 * left zeroed (only root has meaningful aggregates).
 */
inline RankStats reduce_scalar_across_ranks(MPI_Comm comm, double local_value,
                                            int root = 0) {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  std::vector<double> all;
  if (rank == root) all.resize(static_cast<std::size_t>(size));

  MPI_Gather(&local_value, 1, MPI_DOUBLE, rank == root ? all.data() : nullptr, 1,
             MPI_DOUBLE, root, comm);

  RankStats s{};
  s.count = size;
  if (rank != root) return s;

  s.min = all[0];
  s.max = all[0];
  s.sum = 0.0;
  for (double x : all) {
    s.min = std::min(s.min, x);
    s.max = std::max(s.max, x);
    s.sum += x;
  }
  s.mean = s.sum / static_cast<double>(size);
  if (size >= 2) {
    double acc = 0.0;
    for (double x : all) {
      const double d = x - s.mean;
      acc += d * d;
    }
    s.stddev_sample = std::sqrt(acc / static_cast<double>(size - 1));
  }
  return s;
}

/**
 * @brief MPI_MAX reduce of local_value to root (common for reporting slowest rank).
 */
inline double reduce_max_to_root(MPI_Comm comm, double local_value, int root = 0) {
  double out = 0.0;
  MPI_Reduce(&local_value, &out, 1, MPI_DOUBLE, MPI_MAX, root, comm);
  return out;
}

} // namespace profiling
} // namespace pfc

#endif // PFC_KERNEL_PROFILING_MPI_STATS_HPP
