// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cstdlib>
#include <iostream>
#include <tungsten/cpu/tungsten.hpp>

int main(int argc, char *argv[]) {
  try {
    std::cout << std::fixed;
    std::cout.precision(3);
    pfc::ui::App<Tungsten> app(argc, argv);
    return app.main();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
  } catch (...) {
    return EXIT_FAILURE;
  }
}
