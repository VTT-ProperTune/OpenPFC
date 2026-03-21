// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file array_format.hpp
 * @brief String formatting and console display for small arrays (kernel layer)
 */

#ifndef PFC_KERNEL_DETAIL_ARRAY_FORMAT_HPP
#define PFC_KERNEL_DETAIL_ARRAY_FORMAT_HPP

#include <openpfc/kernel/detail/typename.hpp>

#include <array>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace pfc {
namespace detail {

template <typename T, std::size_t D>
std::string array_to_string(const std::array<T, D> &arr) {
  std::ostringstream oss;
  oss << '{';
  for (std::size_t i = 0; i < D; ++i) {
    oss << arr[i];
    if (i != D - 1) oss << ", ";
  }
  oss << '}';
  return oss.str();
}

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
        std::cout << std::setw(9) << std::setprecision(6) << std::fixed << data[idx]
                  << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

template <typename T>
void show(const std::vector<T> &data, const std::array<int, 2> &size,
          [[maybe_unused]] const std::array<int, 2> &offsets) {
  std::cout << size[0] << "×" << size[1] << " Array<2, " << pfc::TypeName<T>::get()
            << ">:" << std::endl;
  for (int i = 0; i < size[0]; ++i) {
    for (int j = 0; j < size[1]; ++j) {
      size_t idx = j * size[0] + i;
      std::cout << std::setw(9) << std::setprecision(6) << std::fixed << data[idx]
                << " ";
    }
    std::cout << std::endl;
  }
}

} // namespace detail
} // namespace pfc

#endif
