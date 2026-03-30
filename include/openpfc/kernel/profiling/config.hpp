// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file config.hpp
 * @brief Compile-time OPENPFC_PROFILING_LEVEL (0 vs >0); see profile_scope_macro.hpp
 *
 * @see ProfilingSession for the primary profiling API
 */

#ifndef PFC_KERNEL_PROFILING_CONFIG_HPP
#define PFC_KERNEL_PROFILING_CONFIG_HPP

#ifndef OPENPFC_PROFILING_LEVEL
#define OPENPFC_PROFILING_LEVEL 2
#endif

#endif // PFC_KERNEL_PROFILING_CONFIG_HPP
