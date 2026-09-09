// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file thin_film_nonlinear.cpp
 * @brief `thin_film_nonlinear` main: `run_thin_film_nonlinear` on the host.
 *
 * Usage: `thin_film_nonlinear CASE.json`. See `thin_film/nonlinear_driver.hpp`
 * for the physics and `src/hip/thin_film_nonlinear.cpp` for the device twin.
 */

#include <iostream>
#include <stdexcept>

#include <mpi.h>

#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <thin_film/nonlinear_driver.hpp>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  int status = 0;
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: thin_film_nonlinear CASE.json");
    status = thin_film::run_thin_film_nonlinear<pfc::HostSpace,
                                                pfc::sim::stacks::SpectralCPUStack>(
        rank, nproc, MPI_COMM_WORLD, argv[1]);
  } catch (const std::exception &e) {
    if (rank == 0) std::cerr << "thin_film_nonlinear: " << e.what() << "\n";
    status = 2;
  }
  MPI_Finalize();
  return status;
}
