// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/frontend/io/binary_writer.hpp>
#include <openpfc/kernel/simulation/results_writer.hpp>

#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char *argv[]) {
  std::cout.setf(std::ios::fixed);
  std::cout.precision(3);
  MPI_Init(&argc, &argv);
  int me = 0;
  int num_ranks = 0;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &num_ranks);

  // MPI-IO collectives in BinaryWriter require every process in the communicator
  // to participate, and this decomposition is only valid for two ranks.
  if (num_ranks != 2) {
    if (me == 0) {
      std::cerr << "This example must be run with exactly 2 MPI processes.\n";
    }
    MPI_Finalize();
    return 1;
  }

  if (me == 0) {
    std::cout << "Writing results with two MPI processes\n";
  }

  {
    pfc::BinaryWriter writer("test_%04d.bin");
    if (me == 0) {
      writer.set_domain({8, 1, 1}, {4, 1, 1}, {0, 0, 0});
      (void)writer.write(5, std::vector<double>{1, 2, 3, 4});
    } else {
      writer.set_domain({8, 1, 1}, {4, 1, 1}, {4, 0, 0});
      (void)writer.write(5, std::vector<double>{5, 6, 7, 8});
    }
  }

  MPI_Barrier(comm);

  if (me == 0) {
    std::ifstream file("test_0005.bin", std::ios::binary);
    if (!file) {
      std::cerr << "Failed to open test_0005.bin for reading\n";
      MPI_Finalize();
      return 1;
    }
    std::vector<double> data(8);
    const auto nbytes = static_cast<std::streamsize>(data.size() * sizeof(double));
    file.read(reinterpret_cast<char *>(data.data()), nbytes);
    if (file.gcount() != nbytes) {
      std::cerr << "Short read from test_0005.bin: got " << file.gcount()
                << " bytes, expected " << nbytes << "\n";
      MPI_Finalize();
      return 1;
    }
    for (int i = 0; i < 8; i++) {
      std::cout << "data[" << i << "] = " << data[i] << '\n';
    }
  }

  MPI_Finalize();
  return 0;
}
