#include <iostream>

#include <openpfc/decomposition.hpp>
#include <openpfc/world.hpp>

using namespace std;
using namespace pfc;

/** \example decomposition.cpp
 *
 * This example demonstrates how to use the World class to create a simulation
 * world and after that, decompose world to smaller domains.
 */
int main(int argc, char *argv[]) {
  // Domain decomposition can be done in a manual manner, just by giving the
  // size of the calculation domain, id number and total number of subdomains:
  int comm_rank = 0, comm_size = 2;
  Decomposition d1(World({32, 4, 4}), comm_rank, comm_size);
  cout << d1 << endl;

  // In practive, we let MPI communicator to decide the number of subdomains.
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);

  World world({32, 4, 4});
  Decomposition d2(world, comm);

  if (comm_rank == 0) {
    cout << d2 << endl;
  }

  MPI_Finalize();
  return 0;
}
