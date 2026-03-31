// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file profile_scope_macro.hpp
 * @brief OPENPFC_PROFILE("path") { … } and PFC_PROFILE_SCOPE (see
 * OPENPFC_PROFILING_LEVEL)
 */

#ifndef PFC_KERNEL_PROFILING_PROFILE_SCOPE_MACRO_HPP
#define PFC_KERNEL_PROFILING_PROFILE_SCOPE_MACRO_HPP

#include <openpfc/kernel/profiling/config.hpp>

#define OPENPFC_PROFILE_CONCAT_INNER_(a, b) a##b
#define OPENPFC_PROFILE_CONCAT_(a, b) OPENPFC_PROFILE_CONCAT_INNER_(a, b)
/// Unique per source line and macro expansion (avoids nested-scope name clashes /
/// -Wshadow).
#define OPENPFC_PROFILE_UID_(pfx)                                                   \
  OPENPFC_PROFILE_CONCAT_(pfx, OPENPFC_PROFILE_CONCAT_(__LINE__, __COUNTER__))

#if OPENPFC_PROFILING_LEVEL <= 0

#define OPENPFC_PROFILE(NAME_STR) if (true)
#define PFC_PROFILE_SCOPE(name) ((void)0)

#else

#include <openpfc/kernel/profiling/region_scope.hpp>

/// TimerOutputs-style block: OPENPFC_PROFILE("my/region") { … }; registers path on
/// first use.
#define OPENPFC_PROFILE(NAME_STR)                                                   \
  if (::pfc::profiling::ProfilingTimedScope OPENPFC_PROFILE_UID_(_openpfc_prof_){   \
          (NAME_STR)};                                                              \
      true)

/// Single-statement scope (unique local name via __LINE__ / __COUNTER__).
#define PFC_PROFILE_SCOPE(name)                                                     \
  ::pfc::profiling::ProfilingTimedScope OPENPFC_PROFILE_UID_(_pfc_profile_scope_) { \
    (name)                                                                          \
  }

#endif

#endif // PFC_KERNEL_PROFILING_PROFILE_SCOPE_MACRO_HPP
