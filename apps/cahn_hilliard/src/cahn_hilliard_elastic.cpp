// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file cahn_hilliard_elastic.cpp
 * @brief Coherent Vegard Cahn–Hilliard on the host Green operator.
 *
 * Usage: `cahn_hilliard_elastic CASE.json`. Host-only (#157).
 */

#include <iostream>
#include <stdexcept>

#include <mpi.h>

#include <cahn_hilliard/elastic_driver.hpp>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  int status = 0;
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: cahn_hilliard_elastic CASE.json");
    status = cahn_hilliard::run_cahn_hilliard_elastic(rank, nproc, MPI_COMM_WORLD,
                                                      argv[1]);
  } catch (const std::exception &e) {
    if (rank == 0) std::cerr << "cahn_hilliard_elastic: " << e.what() << "\n";
    status = 2;
  }
  MPI_Finalize();
  return status;
}
