// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file utils.hpp
 * @brief General utility functions
 *
 * @details
 * This header provides general utility functions used throughout OpenPFC,
 * including string formatting, file operations, and helper functions.
 *
 * Key utilities:
 * - string_format(): Printf-style string formatting in C++
 * - File I/O helpers
 * - String manipulation utilities
 * - MPI-safe operations
 *
 * @code
 * #include <openpfc/utils.hpp>
 *
 * // Format strings safely
 * std::string msg = pfc::utils::string_format("Step %d, time = %.3f", step, time);
 * @endcode
 *
 * @see utils/array_to_string.hpp for array formatting
 * @see utils/show.hpp for debug printing
 * @see utils/typename.hpp for type introspection
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef PFC_UTILS_HPP
#define PFC_UTILS_HPP

#include <memory>
#include <mpi.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pfc {
namespace utils {

// Overload for no format arguments - just return the string as-is
inline std::string string_format(const std::string &str) { return str; }

template <typename... Args>
inline std::string string_format(const std::string &format, Args... args) {
  size_t size =
      snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
  if (size <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  std::unique_ptr<char[]> buf(new char[size]);
  snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(),
                     buf.get() + size - 1); // We don't want the '\0' inside
}

inline std::string format_with_number(const std::string &filename, int increment) {
  if (filename.find('%') != std::string::npos) {
    return utils::string_format(filename, increment);
  } else {
    return filename;
  }
}

template <typename T> size_t sizeof_vec(std::vector<T> &V) {
  return V.size() * sizeof(T);
}

} // namespace utils

namespace mpi {

inline int get_comm_rank(MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

inline int get_comm_size(MPI_Comm comm) {
  int size;
  MPI_Comm_size(comm, &size);
  return size;
}

} // namespace mpi

} // namespace pfc

#endif
