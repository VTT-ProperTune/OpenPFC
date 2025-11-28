// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "Tungsten.hpp"

int main(int argc, char *argv[]) {
  cout << std::fixed;
  cout.precision(3);
  App<Tungsten> app(argc, argv);
  return app.main();
}
