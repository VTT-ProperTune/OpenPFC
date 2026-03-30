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

namespace pfc {
namespace profiling {

/**
 * @brief Format byte count as B, KB, MB, GB, or TB (1024-based scales; labels
 *        match historical OpenPFC memory reports).
 */
inline std::string format_bytes(std::size_t bytes) {
  const double KB = 1024.0;
  const double MB = KB * 1024.0;
  const double GB = MB * 1024.0;
  const double TB = GB * 1024.0;

  std::ostringstream oss;
  oss.precision(2);
  oss << std::fixed;

  if (bytes >= TB) {
    oss << (static_cast<double>(bytes) / TB) << " TB";
  } else if (bytes >= GB) {
    oss << (static_cast<double>(bytes) / GB) << " GB";
  } else if (bytes >= MB) {
    oss << (static_cast<double>(bytes) / MB) << " MB";
  } else if (bytes >= KB) {
    oss << (static_cast<double>(bytes) / KB) << " KB";
  } else {
    oss << bytes << " B";
  }
  return oss.str();
}

} // namespace profiling
} // namespace pfc

#endif // PFC_KERNEL_PROFILING_FORMAT_HPP
