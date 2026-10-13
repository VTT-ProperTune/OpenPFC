// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/runtime/common/mpi_main.hpp>
#include <stdexcept>

int main(int argc, char **argv) {
  return pfc::runtime::mpi_main(argc, argv, [](int, char **, int rank, int ranks) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized || ranks < 1 || rank < 0 || rank >= ranks)
      throw std::runtime_error("MPI startup did not initialize the body");
    std::cout << "Host startup consumer ranks=" << ranks << '\n';
    return EXIT_SUCCESS;
  });
}
