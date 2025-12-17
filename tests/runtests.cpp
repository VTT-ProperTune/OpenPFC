// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp> // Include for Catch::Session
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <openpfc/mpi/worker.hpp>

int main(int argc, char *argv[]) {
  // Initialize MPI using MPI_Worker (handles case where MPI is already initialized)
  // verbose=false to avoid test discovery issues with Catch2
  pfc::MPI_Worker worker(argc, argv, MPI_COMM_WORLD, false);

  // Run Catch2 tests
  int result = Catch::Session().run(argc, argv); // Use Catch::Session

  // MPI_Worker destructor will handle MPI_Finalize if needed
  return result;
}
