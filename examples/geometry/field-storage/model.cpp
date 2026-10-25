// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <array>
#include <cmath>
#include <iostream>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>

int main() {
  // start snippet storage
  const auto domain = pfc::domain::create(pfc::GridSize{{8, 6, 4}},
                                          pfc::PhysicalOrigin{{-1.0, 0.0, 0.0}},
                                          pfc::GridSpacing{{0.25, 0.5, 1.0}});
  const auto box = pfc::Box3i::from_bounds({4, 0, 0}, {7, 5, 3});
  pfc::data::Field<double> compact(domain, box, 0);
  pfc::data::Field<double> padded(domain, box, 1);
  auto polynomial = [](double x, double y, double z) { return 1 + x + y * y + z; };
  compact.apply(polynomial);
  padded.apply(polynomial);
  // end snippet storage

  double mismatch = 0.0;
  padded.for_each_owned([&](int i, int j, int k) {
    mismatch += std::abs(padded(i, j, k) - compact(i, j, k));
  });
  // start snippet components
  pfc::data::Field<std::array<double, 3>> positions(domain, box, 0);
  positions.apply([](double x, double y, double z) { return std::array{x, y, z}; });
  const auto first = positions(0, 0, 0);
  // end snippet components
  std::cout << "owned=" << compact.size() << " padded=" << padded.size()
            << " first_owned_offset=" << padded.idx(0, 0, 0)
            << " first_x=" << first[0] << " mismatch=" << mismatch << '\n';
  return compact.size() == 96 && padded.size() == 288 && padded.idx(0, 0, 0) == 55 &&
                 mismatch == 0.0 && first[0] == 0.0
             ? 0
             : 1;
}
