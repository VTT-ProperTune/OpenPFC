// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/factory/decomposition_factory.hpp"

namespace pfc {

Decomposition make_decomposition(const World &world, MPI_Comm comm) {
  int rank;
  int size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  return Decomposition(world, rank, size);
}

} // namespace pfc
