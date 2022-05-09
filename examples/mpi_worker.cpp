#include <iostream>
#include <pfc/mpi/communicator.hpp>
#include <pfc/mpi/environment.hpp>

int main() {
  pfc::mpi::environment env;
  pfc::mpi::communicator comm;
  std::cout << "I am process " << comm.rank() << " of " << comm.size() << "."
            << std::endl;
  return 0;
}
