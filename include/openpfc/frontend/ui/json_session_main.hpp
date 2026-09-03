// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file json_session_main.hpp
 * @brief Shared `main()` body for JSON-driven session binaries.
 *
 * Loads the JSON/TOML settings file named on the command line, constructs
 * `Session(settings, rank, nproc, comm)`, runs it to `t1`, and prints a
 * one-line completion message on rank 0. Apps may pass a `setup` callable
 * (run once before the session is constructed) to register app-specific
 * catalog entries such as initial conditions or boundary conditions.
 */

#include <cstdlib>
#include <iostream>

#include <mpi.h>

#include <openpfc/frontend/ui/settings_loader.hpp>
#include <openpfc/kernel/simulation/time.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>

namespace pfc::ui {

struct NoSessionSetup {
  constexpr void operator()() const noexcept {}
};

template <class Session, class Setup = NoSessionSetup>
[[nodiscard]] int run_json_session_main(int argc, char *argv[],
                                        const char *done_label,
                                        Setup &&setup = {}) {
  return pfc::runtime::mpi_main(
      argc, argv, [&](int app_argc, char **app_argv, int rank, int nproc) {
        if (app_argc <= 1) {
          if (rank == 0) {
            std::cerr << "Usage: " << app_argv[0] << " <config.json|config.toml>\n";
          }
          return EXIT_FAILURE;
        }
        setup();
        const auto settings = pfc::ui::load_settings_file(app_argv[1]);
        Session session(settings, rank, nproc, MPI_COMM_WORLD);
        session.run();
        if (rank == 0) {
          std::cout << done_label << " done t=" << pfc::time::current(session.time())
                    << '\n';
        }
        return EXIT_SUCCESS;
      });
}

} // namespace pfc::ui
