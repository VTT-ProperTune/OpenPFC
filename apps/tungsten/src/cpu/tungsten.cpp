// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>
#include <tungsten/cpu/tungsten.hpp>

int main(int argc, char *argv[]) {
  std::cout << std::fixed;
  std::cout.precision(3);
  pfc::ui::App<Tungsten> app(argc, argv);
  return app.main();
}
