// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iomanip>
#include <iostream>
#include <numbers>
#include <openpfc/kernel/data/domain.hpp>

int main() {
  std::cout << std::setprecision(17) << "grid,index,x,spacing,upper\n";
  // start snippet spacing
  for (bool periodic : {true, false}) {
    const auto grid = pfc::domain::from_bounds(
        {5, 2, 2}, {0, 0, 0}, {2 * std::numbers::pi, 1, 1}, {periodic, true, true});
    for (int i = 0; i < 5; ++i) {
      const double x = pfc::domain::to_coords(grid, {i, 0, 0})[0];
      std::cout << (periodic ? "periodic" : "closed") << ',' << i << ',' << x << ','
                << pfc::domain::get_spacing(grid)[0] << ','
                << pfc::domain::upper(grid)[0] << '\n';
    }
  }
  // end snippet spacing
}
