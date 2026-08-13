// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <array>
#include <cstddef>
#include <iostream>
#include <openpfc/kernel/data/box3i.hpp>

using namespace pfc;

/**
 * \example 06_multi_index.cpp
 *
 * Index boxes are walked with `pfc::Box3i` and `pfc::for_each_index`.
 * `Box3i::to_linear` maps an (i, j, k) point to the x-fastest linear offset
 * used by OpenPFC fields. This example used to demonstrate `MultiIndex`; that
 * type is gone.
 */
int main() {
  // Inclusive [low, high] box of size 2 on every axis, offset from the origin.
  auto box = Box3i::from_bounds({1, 1, 1}, {2, 2, 2});
  std::cout << box << std::endl;

  long long linear = 0;
  for_each_index(box, [&](const std::array<int, 3> &idx) {
    std::cout << '{' << idx[0] << ", " << idx[1] << ", " << idx[2]
              << "}, linear index = " << linear << std::endl;
    ++linear;
  });
  // Result is
  /*
  {1, 1, 1}, linear index = 0
  {2, 1, 1}, linear index = 1
  {1, 2, 1}, linear index = 2
  {2, 2, 1}, linear index = 3
  {1, 1, 2}, linear index = 4
  {2, 1, 2}, linear index = 5
  {1, 2, 2}, linear index = 6
  {2, 2, 2}, linear index = 7
  */

  std::array<int, 8> data{};
  for_each_index(box, [&](const std::array<int, 3> &idx) {
    const auto [i, j, k] = idx;
    data[static_cast<std::size_t>(box.to_linear(idx))] = 2 * i + j + 3 * k;
  });

  std::cout << "data = [";
  for (const auto &v : data)
    std::cout << v << ((&v != &data.back()) ? ", " : "]\n");

  std::cout << "Linear index of {2, 1, 2} = " << box.to_linear({2, 1, 2})
            << std::endl;

  // A 5x5 slice in the xy-plane (size 1 in z), filled over a 3x3 interior.
  std::array<int, 25> arr{};
  auto slice = Box3i::from_bounds({3, 3, 0}, {7, 7, 0});
  for (int j = 4; j <= 6; ++j) {
    for (int i = 4; i <= 6; ++i) {
      arr[static_cast<std::size_t>(slice.to_linear({i, j, 0}))] = 1;
    }
  }
  for (std::size_t i = 0; i < arr.size(); ++i) {
    std::cout << arr[i] << ' ';
    if ((i + 1) % 5 == 0) {
      std::cout << '\n';
    }
  }
  return 0;
}
