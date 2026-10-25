// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <iostream>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>

int main() {
  // start snippet ownership
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  const int rank = comm.rank();
  int sum = 0;
  MPI_Allreduce(&rank, &sum, 1, MPI_INT, MPI_SUM, comm);
  if (rank == 0)
    std::cout << "ranks,rank_sum\n" << comm.size() << ',' << sum << '\n';
  // end snippet ownership
}
