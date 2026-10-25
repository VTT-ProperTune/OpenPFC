// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
/** @file check_error.hpp
 * @brief Checked vendor-runtime calls with operation context.
 */
#include <stdexcept>
#include <string>
#if defined(OpenPFC_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif
#if defined(OpenPFC_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif
namespace pfc::runtime::gpu {
#if defined(OpenPFC_ENABLE_CUDA)
inline void cuda_check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}
#endif
#if defined(OpenPFC_ENABLE_HIP)
inline void hip_check(hipError_t status, const char *operation) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
}
#endif
} // namespace pfc::runtime::gpu
