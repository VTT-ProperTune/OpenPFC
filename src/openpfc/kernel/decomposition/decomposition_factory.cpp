// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <mpi.h>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/mpi/mpi.hpp>

namespace pfc {

[[nodiscard]] Decomposition make_decomposition(const World &world, int rank,
                                               int num_domains) {
  // For explicit rank/size, create a decomposition for all domains
  // Each process can access its own subdomain using its rank
  (void)rank; // Rank is not needed here as decomposition contains all subdomains
  return decomposition::create(world, num_domains);
}

[[nodiscard]] Decomposition make_decomposition(const World &world, MPI_Comm comm) {
  // Fail closed: these were previously unchecked (audit 4.7). The checked
  // helpers throw on a nonzero MPI error code.
  const int size = pfc::mpi::get_comm_size(comm);

  // Create decomposition with automatic grid selection based on number of processes
  return decomposition::create(world, size);
}

} // namespace pfc
