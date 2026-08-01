// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file deep_copy_gpu.hpp
 * @brief Single-source GPU device-to-device deep_copy (M3)
 *
 * Unified implementation for CUDA and HIP backends using the GPU vendor shim.
 * Replaces runtime/cuda/deep_copy_cuda.hpp and runtime/hip/deep_copy_hip.hpp.
 *
 * Include after openpfc/kernel/execution/deep_copy.hpp when copying between
 * GPU memory space Views.
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <openpfc/kernel/execution/deep_copy.hpp>
#include <openpfc/kernel/execution/view.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <stdexcept>

#if defined(OpenPFC_ENABLE_CUDA)
#include <openpfc/runtime/cuda/memory_space_cuda.hpp>
#elif defined(OpenPFC_ENABLE_HIP)
#include <openpfc/runtime/hip/memory_space_hip.hpp>
#endif

namespace pfc {
namespace detail {

#if defined(OpenPFC_ENABLE_CUDA)

template <typename T, std::size_t Rank, typename L1, typename L2>
void deep_copy_view_to_view_impl(View<T, Rank, L1, CudaSpace> &dst,
                                 const View<T, Rank, L2, CudaSpace> &src) {
  const std::size_t n = dst.size();
  if (src.size() != n) {
    throw std::runtime_error("deep_copy: View size mismatch");
  }
  if (n == 0) return;
  gpuError_t err =
      gpuMemcpy(dst.data(), src.data(), n * sizeof(T), gpuMemcpyDeviceToDevice);
  if (err != gpuSuccess) {
    throw std::runtime_error("deep_copy: GPU device-to-device failed: " +
                             std::string(gpuGetErrorString(err)));
  }
}

#elif defined(OpenPFC_ENABLE_HIP)

template <typename T, std::size_t Rank, typename L1, typename L2>
void deep_copy_view_to_view_impl(View<T, Rank, L1, HipSpace> &dst,
                                 const View<T, Rank, L2, HipSpace> &src) {
  const std::size_t n = dst.size();
  if (src.size() != n) {
    throw std::runtime_error("deep_copy: View size mismatch");
  }
  if (n == 0) return;
  gpuError_t err =
      gpuMemcpy(dst.data(), src.data(), n * sizeof(T), gpuMemcpyDeviceToDevice);
  if (err != gpuSuccess) {
    throw std::runtime_error("deep_copy: GPU device-to-device failed: " +
                             std::string(gpuGetErrorString(err)));
  }
}

#endif

} // namespace detail
} // namespace pfc

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP