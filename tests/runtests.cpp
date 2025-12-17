// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp> // Include for Catch::Session
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <openpfc/mpi/worker.hpp>

int main(int argc, char *argv[]) {
  // Initialize MPI once as a static variable (singleton pattern).
  // This keeps MPI initialized across all Catch2 test runs, avoiding
  // per-test MPI_Init/Finalize overhead that causes 1+ sec delay per test.
  // MPI_Worker already checks MPI_Initialized() internally.
  static pfc::MPI_Worker worker(argc, argv, MPI_COMM_WORLD, false);

  // Run Catch2 tests
  int result = Catch::Session().run(argc, argv);
  return result;
}
