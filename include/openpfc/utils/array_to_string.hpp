// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file array_to_string.hpp
 * @brief Convert std::array to string representation
 *
 * @details
 * This header provides the array_to_string() template function for
 * converting std::array<T, D> to a formatted string representation.
 *
 * The function produces output in the format: {value1, value2, value3}
 * which is useful for debugging, logging, and displaying array contents.
 *
 * @code
 * #include <openpfc/utils/array_to_string.hpp>
 *
 * std::array<int, 3> size = {64, 64, 64};
 * std::string str = pfc::utils::array_to_string(size);
 * // str is "{64, 64, 64}"
 * @endcode
 *
 * @see utils/show.hpp for formatted array display
 * @see utils.hpp for other utility functions
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef PFC_UTILS_ARRAY_TO_STRING_HPP
#define PFC_UTILS_ARRAY_TO_STRING_HPP

#include <array>
#include <cstddef>
#include <sstream>
#include <string>

namespace pfc {
namespace utils {

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

} // namespace utils
} // namespace pfc

#endif
