/*

OpenPFC, a simulation software for the phase field crystal method.
Copyright (C) 2024 VTT Technical Research Centre of Finland Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.

*/

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
void show(const std::vector<T> &data, const std::array<int, 3> &size, const std::array<int, 3> &offsets) {
  std::cout << size[0] << "×" << size[1] << "×" << size[2] << " Array<3, " << pfc::TypeName<T>::get()
            << ">:" << std::endl;
  for (int k = 0; k < size[2]; ++k) {
    std::cout << "[:, :, " << offsets[2] + k << "] = " << std::endl;
    for (int i = 0; i < size[0]; ++i) {
      for (int j = 0; j < size[1]; ++j) {
        int idx = k * size[0] * size[1] + j * size[0] + i;
        std::cout << std::setw(9) << std::setprecision(6) << std::fixed << data[idx] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

template <typename T>
void show(const std::vector<T> &data, const std::array<int, 2> &size, const std::array<int, 2> &offsets) {
  std::cout << size[0] << "×" << size[1] << " Array<2, " << pfc::TypeName<T>::get() << ">:" << std::endl;
  for (int i = 0; i < size[0]; ++i) {
    for (int j = 0; j < size[1]; ++j) {
      size_t idx = j * size[0] + i;
      std::cout << std::setw(9) << std::setprecision(6) << std::fixed << data[idx] << " ";
    }
    std::cout << std::endl;
  }
}

} // namespace utils
} // namespace pfc

#endif
