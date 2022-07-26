#include <iostream>

#include <pfc/decomposition.hpp>

using namespace std;
using namespace PFC;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm, &rank);

  if (rank == 0) {
    cout << Decomposition({32, 4, 4}, comm) << endl;

    // or by default, MPI_COMM_WORLD is used
    cout << Decomposition({32, 4, 4});
  }

  MPI_Finalize();
  return 0;
}
