// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

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
  std::cout << "I am process " << s.mpi.comm.rank() << " of "
            << s.mpi.comm.size() << "." << std::endl;
  return 0;
}
