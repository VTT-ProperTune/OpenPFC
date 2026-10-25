// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iomanip>
#include <iostream>
#include <openpfc/kernel/data/domain.hpp>

int main() {
  // start snippet geometry
  const auto grid =
      pfc::domain::create(pfc::GridSize{{8, 4, 2}}, pfc::PhysicalOrigin{{-1, 2, 10}},
                          pfc::GridSpacing{{.25, .5, 3}});
  std::cout << std::setprecision(17) << "i,j,k,x,y,z,ri,rj,rk\n";
  for (const pfc::Int3 index : {pfc::Int3{0, 0, 0}, {3, 2, 1}, {7, 3, 1}}) {
    const auto point = pfc::domain::to_coords(grid, index);
    const auto recovered = pfc::domain::to_indices(grid, point);
    std::cout << index[0] << ',' << index[1] << ',' << index[2] << ',' << point[0]
              << ',' << point[1] << ',' << point[2] << ',' << recovered[0] << ','
              << recovered[1] << ',' << recovered[2] << '\n';
  }
  // end snippet geometry
}
