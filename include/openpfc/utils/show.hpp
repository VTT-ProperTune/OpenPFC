// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef PFC_UTILS_SHOW
#define PFC_UTILS_SHOW

#include "typename.hpp"
#include <array>
#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>

namespace pfc {
namespace utils {

template <typename T>
void show(const std::vector<T> &data, const std::array<int, 3> &size,
          const std::array<int, 3> &offsets) {
  std::cout << size[0] << "×" << size[1] << "×" << size[2] << " Array<3, "
            << pfc::TypeName<T>::get() << ">:" << std::endl;
  for (int k = 0; k < size[2]; ++k) {
    std::cout << "[:, :, " << offsets[2] + k << "] = " << std::endl;
    for (int i = 0; i < size[0]; ++i) {
      for (int j = 0; j < size[1]; ++j) {
        int idx = k * size[0] * size[1] + j * size[0] + i;
        std::cout << std::setw(9) << std::setprecision(6) << std::fixed
                  << data[idx] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

template <typename T>
void show(const std::vector<T> &data, const std::array<int, 2> &size,
          const std::array<int, 2> &offsets) {
  std::cout << size[0] << "×" << size[1] << " Array<2, "
            << pfc::TypeName<T>::get() << ">:" << std::endl;
  for (int i = 0; i < size[0]; ++i) {
    for (int j = 0; j < size[1]; ++j) {
      size_t idx = j * size[0] + i;
      std::cout << std::setw(9) << std::setprecision(6) << std::fixed
                << data[idx] << " ";
    }
    std::cout << std::endl;
  }
}

} // namespace utils
} // namespace pfc

#endif
