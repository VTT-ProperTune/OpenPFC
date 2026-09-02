// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file aluminum_etd.cpp
 * @brief Alias of production `aluminumNew`: JSON → AluminumETDSession.
 */

#include <cstdlib>
#include <iostream>

#include <aluminum/aluminum_etd_session.hpp>
#include <openpfc/frontend/ui/settings_loader.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>

int main(int argc, char **argv) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        if (app_argc <= 1) {
          if (rank == 0) {
            std::cerr << "Usage: " << app_argv[0] << " <config.json|config.toml>\n";
          }
          return EXIT_FAILURE;
        }
        const auto settings = pfc::ui::load_settings_file(app_argv[1]);
        aluminum::AluminumETDSession session(settings, rank, nproc, MPI_COMM_WORLD);
        session.run();
        if (rank == 0) {
          std::cout << "aluminum_etd done t=" << pfc::time::current(session.time())
                    << '\n';
        }
        return EXIT_SUCCESS;
      });
}
