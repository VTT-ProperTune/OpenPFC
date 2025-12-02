// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef PFC_GPU_TEST_HELPERS_HPP
#define PFC_GPU_TEST_HELPERS_HPP

// Only include CUDA headers if CUDA is enabled at compile time
// Note: This header must compile on AMD systems (no CUDA headers available)
// On AMD systems, OpenPFC_ENABLE_CUDA will not be defined, so this is safe
#if defined(OpenPFC_ENABLE_CUDA)
// CMake only defines OpenPFC_ENABLE_CUDA if CUDA was actually found
// So if we're here, CUDA headers should be available
#include <cuda_runtime.h>
#define PFC_GPU_CUDA_HEADERS_AVAILABLE 1
#else
// CUDA not enabled - safe to compile on AMD systems
#define PFC_GPU_CUDA_HEADERS_AVAILABLE 0
#endif

namespace pfc {
namespace gpu {
namespace test {

/**
 * @brief Check if CUDA is available at runtime
 *
 * Returns true only if:
 * 1. CUDA was enabled at compile time (OpenPFC_ENABLE_CUDA defined)
 * 2. CUDA runtime is available
 * 3. At least one CUDA device is present
 *
 * Safe to call on systems without CUDA - will return false.
 */
inline bool is_cuda_available() {
#if PFC_GPU_CUDA_HEADERS_AVAILABLE
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return (err == cudaSuccess) && (device_count > 0);
#else
  return false;
#endif
}

} // namespace test
} // namespace gpu
} // namespace pfc

#undef PFC_GPU_CUDA_HEADERS_AVAILABLE

#endif
