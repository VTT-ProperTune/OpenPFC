// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>
#include <mpi.h>
#include <openpfc/core/decomposition.hpp>
#include <openpfc/core/world.hpp>
#include <openpfc/factory/decomposition_factory.hpp>

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
  MPI_Init(&argc, &argv);
  // Domain decomposition can be done in a manual manner, just by giving the
  // size of the calculation domain, id number and total number of subdomains:
  int comm_rank = 0, comm_size = 2;
  auto world1 = world::create({32, 4, 4});
  auto decomp1 = decomposition::create(world1, comm_size);
  cout << decomp1 << endl;

  MPI_Finalize();
  return 0;
}
