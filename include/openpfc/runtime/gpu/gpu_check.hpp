// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file gpu_check.hpp
 * @brief Single-source GPU error-check helpers for CUDA and HIP (M3).
 *
 * Provides `pfc::cuda::detail::cuda_check` and/or `pfc::hip::detail::hip_check`
 * depending on the enabled backends. Vendor headers `cuda_check.hpp` /
 * `hip_check.hpp` are thin includes of this file so existing call sites keep
 * compiling.
 *
 * Per-tag helpers call the native runtime (not `gpu_api.hpp`) so a CUDA+HIP
 * co-enabled translation unit can use both. `GPU_CHECK` in `gpu_api.hpp` is
 * the single-backend macro for new `runtime/gpu/` sources.
 *
 * @see runtime/gpu/gpu_api.hpp
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <stdexcept>
#include <string>

#if defined(OpenPFC_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif
#if defined(OpenPFC_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace pfc::detail {

inline void throw_gpu_status(const char *what, const char *err) {
  throw std::runtime_error(std::string(what) + ": " + err);
}

} // namespace pfc::detail

#if defined(OpenPFC_ENABLE_CUDA)
namespace pfc::cuda::detail {

inline void cuda_check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    pfc::detail::throw_gpu_status(what, cudaGetErrorString(e));
  }
}

} // namespace pfc::cuda::detail
#endif

#if defined(OpenPFC_ENABLE_HIP)
namespace pfc::hip::detail {

inline void hip_check(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    pfc::detail::throw_gpu_status(what, hipGetErrorString(e));
  }
}

} // namespace pfc::hip::detail
#endif

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
