// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file show.hpp
 * @brief Pretty-print 3D arrays to console
 *
 * @details
 * This header provides the show() template function for pretty-printing
 * 3D arrays stored in std::vector with size and offset information.
 *
 * The function displays:
 * - Array dimensions and type
 * - Slice-by-slice output ([:, :, k] format)
 * - Formatted values with fixed precision
 * - Optional offset information for distributed arrays
 *
 * This is particularly useful for debugging field data and visualization
 * of small computational domains.
 *
 * @code
 * #include <openpfc/utils/show.hpp>
 *
 * std::vector<double> data(4*4*4, 1.0);
 * pfc::utils::show(data, {4, 4, 4}, {0, 0, 0});
 * // Prints formatted 3D array slices
 * @endcode
 *
 * @see utils/array_to_string.hpp for simple array formatting
 * @see utils/typename.hpp for type name display
 *
 * @author OpenPFC Development Team
 * @date 2025
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
          const std::array<int, 2> &offsets) {
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

} // namespace utils
} // namespace pfc

#endif
