// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp> // Include for Catch::Session
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <mpi.h>

int main(int argc, char *argv[]) {
  // Initialize MPI
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cerr << "MPI initialization failed" << std::endl;
    return 1;
  }

  // Run Catch2 tests
  int result = Catch::Session().run(argc, argv); // Use Catch::Session

  // Finalize MPI
  MPI_Finalize();
  return result;
}
