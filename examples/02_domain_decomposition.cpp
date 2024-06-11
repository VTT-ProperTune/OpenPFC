/*

OpenPFC, a simulation software for the phase field crystal method.
Copyright (C) 2024 VTT Technical Research Centre of Finland Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.

*/

#include <iostream>
#include <mpi.h>
#include <openpfc/decomposition.hpp>
#include <openpfc/world.hpp>

using namespace std;
using namespace pfc;

/** \example 02_domain_decomposition.cpp
 *
 * Once the World object has been defined, the next step is to determine how the
 * world will be divided among different MPI processes. In OpenPFC, the
 * calculation area is partitioned into smaller parts using the Decomposition
 * class. This class utilizes the excellent functionality provided by HeFFTe.
 * The partitioning is done in such a way that minimizes the surface area
 * between the regions, which in turn minimizes the MPI communication required
 * during FFT calculations. In the example, we can see how the domain
 * decomposition can be done "manually". However, in practice, it is most
 * effective to divide the calculation area into the same number of parts as the
 * number of computing nodes allocated from the HPC cluster.
 *
 * This example demonstrates how to use the World class to create a simulation
 * world and after that, decompose world to smaller domains using Decomposition
 * class.
 */
int main(int argc, char *argv[]) {
  // Domain decomposition can be done in a manual manner, just by giving the
  // size of the calculation domain, id number and total number of subdomains:
  int comm_rank = 0, comm_size = 2;
  World world1({32, 4, 4});
  Decomposition decomp1(world1, comm_rank, comm_size);
  cout << decomp1 << endl;

  // In practive, we let MPI communicator to decide the number of subdomains.
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  World world2({32, 4, 4});
  Decomposition decomp2(world2, comm);
  if (comm_rank == 0) cout << decomp2 << endl;

  // By default, MPI_COMM_WORLD is used, so the above example can be simplified:
  cout << Decomposition(world2) << endl;

  MPI_Finalize();
  return 0;
}
