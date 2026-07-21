// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file cuda_check.hpp
 * @brief Shared CUDA error check helper for runtime/cuda headers
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA)

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace pfc {
namespace cuda {
namespace detail {

inline void cuda_check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

} // namespace detail
} // namespace cuda
} // namespace pfc

#endif // OpenPFC_ENABLE_CUDA
