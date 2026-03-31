// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file names.hpp
 * @brief Canonical profiling region path constants (must appear in catalog)
 */

#ifndef PFC_KERNEL_PROFILING_NAMES_HPP
#define PFC_KERNEL_PROFILING_NAMES_HPP

#include <string_view>

namespace pfc::profiling {

inline constexpr std::string_view kProfilingRegionFft = "fft";
inline constexpr std::string_view kProfilingRegionCommunication = "communication";
inline constexpr std::string_view kProfilingRegionGradient = "gradient";

} // namespace pfc::profiling

#endif // PFC_KERNEL_PROFILING_NAMES_HPP
