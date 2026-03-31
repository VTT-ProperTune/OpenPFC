// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "Aluminum.hpp"
#include "SeedGridFCC.hpp"
#include "SlabFCC.hpp"

#include <iostream>

int main(int argc, char *argv[]) {
  try {
    std::cout << std::fixed;
    std::cout.precision(3);
    pfc::ui::register_field_modifier<SeedGridFCC>("seed_grid_fcc");
    pfc::ui::register_field_modifier<SlabFCC>("slab_fcc");
    pfc::ui::App<Aluminum> app(argc, argv);
    return app.main();
  } catch (...) {
    return 1;
  }
}
