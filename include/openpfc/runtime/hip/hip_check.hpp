// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file hip_check.hpp
 * @brief Shared HIP error check helper for runtime/hip headers
 */

#pragma once

#if defined(OpenPFC_ENABLE_HIP)

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace pfc {
namespace hip {
namespace detail {

inline void hip_check(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
  }
}

} // namespace detail
} // namespace hip
} // namespace pfc

#endif // OpenPFC_ENABLE_HIP
