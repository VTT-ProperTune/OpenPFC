// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file deep_copy_gpu.hpp
 * @brief Single-source GPU device-to-device `deep_copy` for CUDA and HIP (M3).
 *
 * Overloads `pfc::detail::deep_copy_view_to_view_impl` for `CudaSpace` and/or
 * `HipSpace`. Vendor headers `deep_copy_cuda.hpp` / `deep_copy_hip.hpp` are
 * thin includes of this file.
 *
 * @see kernel/execution/deep_copy.hpp
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <cstddef>
#include <stdexcept>
#include <string>

#include <openpfc/kernel/execution/deep_copy.hpp>
#include <openpfc/kernel/execution/view.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>

#if defined(OpenPFC_ENABLE_CUDA)
#include <cuda_runtime.h>
#include <openpfc/runtime/cuda/memory_space_cuda.hpp>
#include <openpfc/runtime/cuda/view_cuda.hpp>
#endif
#if defined(OpenPFC_ENABLE_HIP)
#include <hip/hip_runtime.h>
#include <openpfc/runtime/hip/memory_space_hip.hpp>
#include <openpfc/runtime/hip/view_hip.hpp>
#endif

namespace pfc::detail {

template <typename T, std::size_t Rank, typename L1, typename L2, typename Space,
          typename Alloc>
void gpu_deep_copy_d2d(View<T, Rank, L1, Space> &dst,
                       const View<T, Rank, L2, Space> &src) {
  const std::size_t n = dst.size();
  if (src.size() != n) {
    throw std::runtime_error("deep_copy: View size mismatch");
  }
  if (n == 0) {
    return;
  }
  auto err = Alloc::memcpy_d2d(dst.data(), src.data(), n * sizeof(T));
  if (err != Alloc::success) {
    throw std::runtime_error(std::string("deep_copy: ") + Alloc::kind +
                             " device-to-device failed");
  }
}

#if defined(OpenPFC_ENABLE_CUDA)
struct CudaD2D {
  using error_t = cudaError_t;
  static constexpr error_t success = cudaSuccess;
  static constexpr const char *kind = "CUDA";
  static error_t memcpy_d2d(void *dst, const void *src, std::size_t bytes) {
    return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
  }
};

template <typename T, std::size_t Rank, typename L1, typename L2>
void deep_copy_view_to_view_impl(View<T, Rank, L1, CudaSpace> &dst,
                                 const View<T, Rank, L2, CudaSpace> &src) {
  gpu_deep_copy_d2d<T, Rank, L1, L2, CudaSpace, CudaD2D>(dst, src);
}
#endif

#if defined(OpenPFC_ENABLE_HIP)
struct HipD2D {
  using error_t = hipError_t;
  static constexpr error_t success = hipSuccess;
  static constexpr const char *kind = "HIP";
  static error_t memcpy_d2d(void *dst, const void *src, std::size_t bytes) {
    return hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice);
  }
};

template <typename T, std::size_t Rank, typename L1, typename L2>
void deep_copy_view_to_view_impl(View<T, Rank, L1, HipSpace> &dst,
                                 const View<T, Rank, L2, HipSpace> &src) {
  gpu_deep_copy_d2d<T, Rank, L1, L2, HipSpace, HipD2D>(dst, src);
}
#endif

} // namespace pfc::detail

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
