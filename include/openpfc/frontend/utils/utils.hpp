// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
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
 * #include <openpfc/frontend/utils/utils.hpp>
 *
 * // Format strings safely
 * std::string msg = pfc::utils::string_format("Step %d, time = %.3f", step, time);
 * @endcode
 *
 * @see utils/array_to_string.hpp for array formatting
 * @see utils/show.hpp for debug printing
 * @see utils/typename.hpp for type introspection
 * @see utils/field_iteration.hpp for inbox iteration helpers
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef PFC_UTILS_HPP
#define PFC_UTILS_HPP

#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <regex>

// pfc::mpi::get_comm_rank and get_comm_size are in kernel/mpi/mpi.hpp.
// Include it when you need MPI rank/size with a specific communicator.
// Frontend code may use: #include <openpfc/kernel/mpi/mpi.hpp>

namespace pfc::utils {

// Overload for no format arguments - just return the string as-is
inline std::string string_format(const std::string &str) { return str; }

template <typename... Args>
inline std::string string_format(const std::string &format, Args... args) {
  const int n = std::snprintf(nullptr, 0, format.c_str(), args...);
  if (n < 0) {
    throw std::runtime_error("Error during formatting.");
  }
  const size_t size = static_cast<size_t>(n) + 1;
  std::vector<char> buf(size);
  snprintf(buf.data(), size, format.c_str(), args...);
  // Braced init avoids repeating std::string in the return
  // (modernize-return-braced-init-list).
  return {buf.data(), buf.data() + size - 1}; // omit trailing '\0'
}

/**
 * @brief Validate a filename pattern string for safe formatting
 *
 * @details
 * Validates that a filename pattern string is safe to use with printf-style formatting.
 * The pattern must contain at most one `%%` conversion specifier, and if present,
 * it must be an integer-only specifier (`%%d` with optional flags/width).
 *
 * Valid patterns may contain:
 * - No `%%` specifier at all (returns unchanged)
 * - A single `%%d` integer specifier with optional flags (`-+ 0`) and width (`*` or digits)
 *
 * Invalid and dangerous patterns that are rejected:
 * - `%%s` (string dereference - arbitrary pointer read)
 * - `%%n` (write count to pointer - arbitrary memory write)
 * - `%%f`, `%%e`, `%%g`, etc. (floating-point specifiers)
 * - `%%x`, `%%p`, etc. (hex/pointer specifiers)
 * - Multiple `%%` specifiers of any kind
 *
 * This validation prevents format-string vulnerabilities where user-supplied
 * filename patterns from CLI arguments or JSON configs could cause undefined behavior.
 *
 * @param pattern Filename pattern string to validate
 *
 * @throws std::invalid_argument if the pattern contains dangerous conversion
 *         specifiers or multiple `%%` sequences
 *
 * @see format_with_number
 */
inline void validate_filename_pattern(const std::string &pattern) {
  static const std::regex valid_pattern(R"(^[^%]*%[-+ 0]*\*?[0-9]*d[^%]*$)");
  if (!std::regex_match(pattern, valid_pattern)) {
    throw std::invalid_argument(
        std::string("format_with_number: invalid filename pattern '") + pattern +
        "'. Must contain at most one %% integer specifier (%%d with optional flags/width). "
        "Dangerous specifiers like %%s, %%n, %%f are not allowed.");
  }
}

/**
 * @brief Format a filename pattern with an increment number
 *
 * @details
 * Formats a filename pattern string by replacing a single `%%d` integer conversion
 * specifier with the provided increment value. If no `%%` specifier is present,
 * the filename is returned unchanged.
 *
 * The format string is validated before formatting to prevent format-string
 * vulnerabilities. Only integer `%%d` specifiers with optional flags (`-+ 0`) and
 * width (`*` or digits) are allowed; all other specifiers are rejected for safety.
 *
 * Common usage patterns for numbered output files:
 * - `output_%%d.dat` → `output_0.dat`, `output_1.dat`, ...
 * - `frame_%%04d.vti` → `frame_0000.vti`, `frame_0001.vti`, ...
 * - `result_%%-5d.bin` → `result_0    .bin`, `result_1    .bin`, ...
 *
 * @param filename Filename pattern string, optionally containing a single `%%d` specifier
 * @param increment Integer value to substitute for the `%%d` specifier
 *
 * @return Formatted filename string with the increment value substituted
 *
 * @throws std::invalid_argument if the filename pattern contains dangerous conversion
 *         specifiers or multiple `%%` sequences
 *
 * @see validate_filename_pattern
 */
inline std::string format_with_number(const std::string &filename, int increment) {
  if (filename.find('%') != std::string::npos) {
    validate_filename_pattern(filename);
    return string_format(filename, increment);
  }
  return filename;
}

template <typename T> size_t sizeof_vec(const std::vector<T> &V) {
  return V.size() * sizeof(T);
}

} // namespace pfc::utils

#endif
