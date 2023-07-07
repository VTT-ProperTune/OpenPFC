#ifndef PFC_UTILS_ARRAY_TO_STRING_HPP
#define PFC_UTILS_ARRAY_TO_STRING_HPP

#include <array>
#include <cstddef>
#include <sstream>
#include <string>

namespace pfc {
namespace utils {

template <typename T, std::size_t D> std::string array_to_string(const std::array<T, D> &arr) {
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
