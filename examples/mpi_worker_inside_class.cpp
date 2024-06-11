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
#include <openpfc/mpi.hpp>

namespace mpi = pfc::mpi;

struct mpi_worker {
  mpi::environment env;
  mpi::communicator comm;
};

class simulation {
public:
  mpi_worker mpi;
};

int main() {
  simulation s;
  std::cout << "I am process " << s.mpi.comm.rank() << " of " << s.mpi.comm.size() << "." << std::endl;
  return 0;
}
