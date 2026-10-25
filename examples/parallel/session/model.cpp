// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <iostream>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>

// start snippet session
struct Simulation {
  pfc::mpi::environment environment;
  pfc::mpi::communicator world;
  pfc::mpi::communicator private_comm = world.duplicate();
};
// end snippet session

int main() {
  int rank = -1, ranks = 0, sum = 0;
  // start snippet lifetime
  {
    Simulation simulation;
    rank = simulation.private_comm.rank();
    ranks = simulation.private_comm.size();
    MPI_Allreduce(&rank, &sum, 1, MPI_INT, MPI_SUM, simulation.private_comm);
  } // private_comm and world are destroyed before environment finalizes MPI.
  int finalized = 0;
  MPI_Finalized(&finalized); // This query is permitted after MPI_Finalize.
  if (rank == 0)
    std::cout << "ranks,rank_sum,finalized\n"
              << ranks << ',' << sum << ',' << finalized << '\n';
  // end snippet lifetime
  return finalized ? 0 : 1;
}
