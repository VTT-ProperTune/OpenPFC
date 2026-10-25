// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>
#include <openpfc/kernel/data/domain.hpp>
#include <type_traits>

// start snippet arguments
pfc::Domain make_domain(pfc::GridSize size, pfc::PhysicalOrigin origin,
                        pfc::GridSpacing spacing) {
  return pfc::domain::create(size, origin, spacing);
}
static_assert(std::is_invocable_v<decltype(&make_domain), pfc::GridSize,
                                  pfc::PhysicalOrigin, pfc::GridSpacing>);
static_assert(!std::is_invocable_v<decltype(&make_domain), pfc::GridSize,
                                   pfc::GridSpacing, pfc::PhysicalOrigin>);
// end snippet arguments

int main() {
  // start snippet construct
  const auto grid =
      make_domain(pfc::GridSize{{8, 4, 2}}, pfc::PhysicalOrigin{{-1, -2, 0}},
                  pfc::GridSpacing{{.25, 1, 2}});
  const auto center = pfc::domain::center(grid);
  std::cout << "points,cx,cy,cz\n"
            << pfc::domain::get_total_size(grid) << ',' << center[0] << ','
            << center[1] << ',' << center[2] << '\n';
  // end snippet construct
}
