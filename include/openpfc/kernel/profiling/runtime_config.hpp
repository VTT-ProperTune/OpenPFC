// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file runtime_config.hpp
 * @brief Optional runtime profiling switches (reserved / documentation)
 *
 * Primary configuration is via JSON `profiling` block and ProfilingExportOptions.
 */

#ifndef PFC_KERNEL_PROFILING_RUNTIME_CONFIG_HPP
#define PFC_KERNEL_PROFILING_RUNTIME_CONFIG_HPP

namespace pfc {
namespace profiling {

struct ProfilingRuntimeConfig {
  /// When true, include RSS and heap byte columns in each frame (if session used).
  bool sample_memory = false;
};

} // namespace profiling
} // namespace pfc

#endif // PFC_KERNEL_PROFILING_RUNTIME_CONFIG_HPP
