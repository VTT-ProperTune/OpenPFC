#include <iostream>
#include <openpfc/mpi/communicator.hpp>
#include <openpfc/mpi/environment.hpp>

int main() {
  pfc::mpi::environment env;
  pfc::mpi::communicator comm;
  std::cout << "I am process " << comm.rank() << " of " << comm.size() << "."
            << std::endl;
  return 0;
}
