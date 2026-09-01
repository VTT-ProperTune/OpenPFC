// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file tungsten_app_main.hpp
 * @brief Shared `main()` body for CPU/CUDA/HIP tungsten ETD session drivers
 */

#ifndef TUNGSTEN_COMMON_APP_MAIN_HPP
#define TUNGSTEN_COMMON_APP_MAIN_HPP

#include <cstdlib>
#include <iostream>

#include <openpfc/frontend/ui/settings_loader.hpp>
#include <openpfc/kernel/simulation/time.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>

namespace tungsten {

/**
 * @brief Load JSON/TOML, construct @p Session, run to `t1`.
 */
template <typename Session>
[[nodiscard]] int run_tungsten_etd_main(int argc, char *argv[],
                                        const char *done_label) {
  return pfc::runtime::mpi_main(
      argc, argv, [done_label](int app_argc, char **app_argv, int rank, int nproc) {
        if (app_argc <= 1) {
          if (rank == 0) {
            std::cerr << "Usage: " << app_argv[0] << " <config.json|config.toml>\n";
          }
          return EXIT_FAILURE;
        }
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

} // namespace tungsten

#endif
