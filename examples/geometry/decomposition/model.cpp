// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>

int main() {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    // start snippet partition
    const auto domain = pfc::domain::create({16, 8, 4});
    const auto partition = pfc::decomposition::create(domain, comm.size());
    const auto box = pfc::decomposition::local_box(partition, comm.rank());
    const long long local_cells =
        static_cast<long long>(box.size[0]) * box.size[1] * box.size[2];
    long long global_cells = 0;
    MPI_Allreduce(&local_cells, &global_cells, 1, MPI_LONG_LONG, MPI_SUM, comm);
    // end snippet partition

    // Gather boxes for display; field values themselves remain distributed.
    const std::array<int, 6> local{box.low[0],  box.low[1],  box.low[2],
                                   box.size[0], box.size[1], box.size[2]};
    std::vector<int> boxes(static_cast<std::size_t>(6 * comm.size()));
    MPI_Gather(local.data(), 6, MPI_INT, boxes.data(), 6, MPI_INT, 0, comm);
    if (global_cells != 16 * 8 * 4) return 1;
    if (comm.rank() == 0) {
      std::filesystem::create_directories("results");
      std::ofstream output("results/decomposition.csv");
      output << "rank,x0,y0,z0,nx,ny,nz\n";
      for (int rank = 0; rank < comm.size(); ++rank) {
        output << rank;
        for (int axis = 0; axis < 6; ++axis) output << ',' << boxes[6 * rank + axis];
        output << '\n';
      }
      if (!output) throw std::runtime_error("Cannot write decomposition.csv");
      std::cout << comm.size() << " ranks own " << global_cells << " cells\n";
    }
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
