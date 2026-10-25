// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <array>
#include <iostream>
#include <openpfc/kernel/data/box3i.hpp>

int main() {
  // start snippet indexing
  const auto box = pfc::Box3i::from_bounds({1, 1, 1}, {2, 2, 2});
  std::array<int, 8> values{};
  pfc::for_each_index(box, [&](const auto &index) {
    const auto [i, j, k] = index;
    const auto offset = box.to_linear(index);
    values[offset] = 2 * i + j + 3 * k;
    std::cout << i << ',' << j << ',' << k << ',' << offset << ',' << values[offset]
              << '\n';
  });
  // end snippet indexing

  // Independent checks of the first and last entries, and an interior stride.
  return values.front() == 6 && values.back() == 12 && box.to_linear({2, 1, 2}) == 5
             ? 0
             : 1;
}
