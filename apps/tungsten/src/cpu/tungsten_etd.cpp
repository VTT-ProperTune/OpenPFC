// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file tungsten_etd.cpp
 * @brief M8 A/B CPU binary: JSON → TungstenEtdSession (mean-field ETD).
 *
 * Gen-1 `tungsten` (`App<Tungsten>`) remains. Same config path; no writers yet.
 */

#include <cstdlib>
#include <iostream>

#include <openpfc/frontend/ui/settings_loader.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>
#include <tungsten/tungsten_etd_session.hpp>

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
        tungsten::TungstenEtdSession session(settings, rank, nproc,
                                             MPI_COMM_WORLD);
        session.run();
        if (rank == 0) {
          std::cout << "tungsten_etd done t=" << pfc::time::current(session.time())
                    << '\n';
        }
        return EXIT_SUCCESS;
      });
}
