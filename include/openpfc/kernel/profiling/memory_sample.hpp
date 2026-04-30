// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file memory_sample.hpp
 * @brief Optional process RSS sampling (Linux)
 *
 * Reads VmRSS from /proc/self/status when available. On other platforms returns
 * 0. This reflects resident set size for the process, not just OpenPFC heap.
 */

#ifndef PFC_KERNEL_PROFILING_MEMORY_SAMPLE_HPP
#define PFC_KERNEL_PROFILING_MEMORY_SAMPLE_HPP

#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>

namespace pfc::profiling {

/**
 * @brief Best-effort resident set size in bytes; 0 if unknown.
 */
inline std::size_t try_read_process_rss_bytes() noexcept {
#if defined(__linux__)
  std::ifstream status("/proc/self/status");
  if (!status) {
    return 0;
  }
  std::string line;
  while (std::getline(status, line)) {
    if (!line.starts_with("VmRSS:")) {
      continue;
    }
    std::istringstream iss(line);
    std::string label;
    std::size_t kb = 0;
    std::string unit;
    if (iss >> label >> kb >> unit) {
      return kb * 1024;
    }
    return 0;
  }
#endif
  return 0;
}

} // namespace pfc::profiling

#endif // PFC_KERNEL_PROFILING_MEMORY_SAMPLE_HPP
