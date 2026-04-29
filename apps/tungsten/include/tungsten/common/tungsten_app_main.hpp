// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file tungsten_app_main.hpp
 * @brief Shared `main()` body for CPU/CUDA/HIP Tungsten driver executables
 */

#ifndef TUNGSTEN_COMMON_APP_MAIN_HPP
#define TUNGSTEN_COMMON_APP_MAIN_HPP

#include <cstdlib>
#include <iostream>
#include <openpfc/frontend/ui/app.hpp>

namespace tungsten {

/**
 * @brief Run `pfc::ui::App<AppModel>` with common console formatting and error
 * handling.
 */
template <typename AppModel>
[[nodiscard]] int run_tungsten_app_main(int argc, char *argv[]) {
  try {
    std::cout << std::fixed;
    std::cout.precision(3);
    pfc::ui::App<AppModel> app(argc, argv);
    return app.main();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
  }
}

} // namespace tungsten

#endif
