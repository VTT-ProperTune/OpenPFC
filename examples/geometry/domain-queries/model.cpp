// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iomanip>
#include <iostream>
#include <openpfc/kernel/data/domain.hpp>

int main() {
  std::cout << std::setprecision(17)
            << "periodic,points,dimension,lx,ly,lz,volume\n";
  // start snippet queries
  for (bool periodic : {true, false}) {
    const auto grid = pfc::domain::create(
        pfc::GridSize{{5, 3, 2}}, pfc::PhysicalOrigin{{-1, 2, 0}},
        pfc::GridSpacing{{.5, 2, 3}}, {periodic, true, true});
    const auto length = pfc::domain::extent(grid);
    std::cout << periodic << ',' << pfc::domain::get_total_size(grid) << ','
              << pfc::domain::dimensionality(grid) << ',' << length[0] << ','
              << length[1] << ',' << length[2] << ','
              << pfc::domain::physical_volume(grid) << '\n';
  }
  // end snippet queries
}
