// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
#include <stdexcept>

namespace pfc::mpi {

/**
 * @brief Collective logical AND: true only if every rank reports success.
 * @note Every communicator member must call in the same collective order.
 */
[[nodiscard]] inline bool all_ranks_succeeded(bool local_success, MPI_Comm comm) {
  const int local = local_success ? 1 : 0;
  int global = 0;
  throw_on_mpi_error(MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_LAND, comm),
                     "MPI_Allreduce LAND in all_ranks_succeeded");
  return global != 0;
}

/**
 * @brief Throw on every rank if any rank reports failure.
 * @param local_success Rank-local acceptance, evaluated before this call.
 * @param comm Participating communicator; this call is collective.
 * @param message Error description to throw locally on collective rejection.
 * @throws std::runtime_error If any member rejects, or the collective fails.
 * @note Does not catch earlier exceptions or roll back caller-owned state.
 */
inline void require_all_ranks(bool local_success, MPI_Comm comm,
                              const char *message) {
  if (!all_ranks_succeeded(local_success, comm)) throw std::runtime_error(message);
}

} // namespace pfc::mpi
