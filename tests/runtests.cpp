// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/mpi/worker.hpp>

#include <string_view>

int main(int argc, char *argv[]) {
  // CMake catch_discover_tests and --list-tests must not MPI_Init: discovery
  // runs the binary without mpiexec and times out if MPI start is slow.
  bool list_only = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg{argv[i]};
    if (arg.rfind("--list-", 0) == 0) {
      list_only = true;
      break;
    }
  }
  if (!list_only) {
    // Initialize MPI once (singleton). Avoids per-test MPI_Init/Finalize.
    static pfc::MPI_Worker worker(argc, argv, MPI_COMM_WORLD, false);
  }

  return Catch::Session().run(argc, argv);
}
