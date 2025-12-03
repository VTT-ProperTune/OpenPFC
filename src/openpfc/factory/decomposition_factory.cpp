// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/factory/decomposition_factory.hpp"
#include "openpfc/core/decomposition.hpp"
#include <mpi.h>

namespace pfc {

Decomposition make_decomposition(const World &world, int rank, int num_domains) {
  // For explicit rank/size, create a decomposition for all domains
  // Each process can access its own subdomain using its rank
  (void)rank; // Rank is not needed here as decomposition contains all subdomains
  return decomposition::create(world, num_domains);
}

Decomposition make_decomposition(const World &world, MPI_Comm comm) {
  int rank = 0;
  int size = 1;

  // Get MPI rank and size from communicator
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // Create decomposition with automatic grid selection based on number of processes
  return decomposition::create(world, size);
}

} // namespace pfc
