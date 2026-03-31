// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file format.hpp
 * @brief Human-readable byte formatting for profiling and memory reports
 */

#ifndef PFC_KERNEL_PROFILING_FORMAT_HPP
#define PFC_KERNEL_PROFILING_FORMAT_HPP

#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>

namespace pfc::profiling {

/**
 * @brief Format byte count as B, KB, MB, GB, or TB (1024-based scales; labels
 *        match historical OpenPFC memory reports).
 */
inline std::string format_bytes(std::size_t bytes) {
  const double KB = 1024.0;
  const double MB = KB * 1024.0;
  const double GB = MB * 1024.0;
  const double TB = GB * 1024.0;
  const auto b = static_cast<double>(bytes);

  std::ostringstream oss;
  oss.precision(2);
  oss << std::fixed;

  if (b >= TB) {
    oss << (b / TB) << " TB";
  } else if (b >= GB) {
    oss << (b / GB) << " GB";
  } else if (b >= MB) {
    oss << (b / MB) << " MB";
  } else if (b >= KB) {
    oss << (b / KB) << " KB";
  } else {
    oss << bytes << " B";
  }
  return oss.str();
}

} // namespace pfc::profiling

#endif // PFC_KERNEL_PROFILING_FORMAT_HPP
