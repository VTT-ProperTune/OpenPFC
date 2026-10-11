// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <chrono>
#include <mpi.h>
#include <openpfc/solvers/multigrid.hpp>
namespace pfc::solvers {
/** Explicit correctness-first MPI adapter with a replicated host hierarchy.
 * owned_indices describes unique ownership of every global degree of freedom.
 * Every application gathers owned values, runs the same hierarchy per rank and
 * returns owned corrections. Fine operator actions may use independent halo
 * exchange. Memory/setup are replicated and this is not a scalable hierarchy.
 */
class ReplicatedMultigrid {
  const Multigrid &hierarchy_;
  MPI_Comm comm_;
  std::vector<int> owned_, counts_, offsets_, indices_;

public:
  double communication_seconds = 0;
  int gathers = 0;
  ReplicatedMultigrid(const Multigrid &hierarchy, std::vector<int> owned,
                      MPI_Comm comm)
      : hierarchy_(hierarchy), comm_(comm), owned_(std::move(owned)) {
    if (comm == MPI_COMM_NULL)
      throw std::invalid_argument("MPI adapter requires communicator");
    int ranks = 0;
    MPI_Comm_size(comm, &ranks);
    counts_.resize(ranks);
    offsets_.resize(ranks);
    int local = owned_.size();
    MPI_Allgather(&local, 1, MPI_INT, counts_.data(), 1, MPI_INT, comm);
    int total = 0;
    for (int r = 0; r < ranks; ++r) {
      offsets_[r] = total;
      total += counts_[r];
    }
    indices_.resize(total);
    MPI_Allgatherv(owned_.data(), local, MPI_INT, indices_.data(), counts_.data(),
                   offsets_.data(), MPI_INT, comm);
    const int n = hierarchy.operator_at(0).size();
    auto sorted = indices_;
    std::sort(sorted.begin(), sorted.end());
    if (total != n)
      throw std::invalid_argument("MPI hierarchy needs complete ownership");
    for (int i = 0; i < n; ++i)
      if (sorted[i] != i)
        throw std::invalid_argument("MPI hierarchy ownership is not unique");
  }
  void operator()(const std::vector<double> &r, std::vector<double> &z,
                  int iteration = 0) {
    int valid = r.size() == owned_.size() ? 1 : 0, all_valid = 0;
    MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN, comm_);
    if (!all_valid)
      throw std::invalid_argument("MPI hierarchy vector ownership mismatch");
    std::vector<double> gathered(indices_.size()), global(indices_.size()),
        correction;
    auto t = std::chrono::steady_clock::now();
    MPI_Allgatherv(r.data(), r.size(), MPI_DOUBLE, gathered.data(), counts_.data(),
                   offsets_.data(), MPI_DOUBLE, comm_);
    communication_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
    ++gathers;
    for (std::size_t i = 0; i < indices_.size(); ++i)
      global[indices_[i]] = gathered[i];
    hierarchy_(global, correction, iteration);
    z.resize(owned_.size());
    for (std::size_t i = 0; i < owned_.size(); ++i) z[i] = correction[owned_[i]];
  }
};
} // namespace pfc::solvers
