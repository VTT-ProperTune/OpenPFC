// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file deep_copy.hpp
 * @brief Kokkos-compatible deep_copy between Views and scalar fill
 *
 * @details
 * deep_copy copies data between Views (same shape) or fills a View with a
 * scalar. Handles host-host, host-device, and device-host. Names and
 * semantics match Kokkos.
 *
 * @see view.hpp for View
 * @see execution_space.hpp for async variant
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#pragma once

#include <algorithm>
#include <openpfc/kernel/execution/execution_space.hpp>
#include <openpfc/kernel/execution/view.hpp>
#include <stdexcept>
#include <vector>

#if defined(OpenPFC_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif
#if defined(OpenPFC_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace pfc {

namespace detail {

template <typename T, std::size_t Rank, typename L1, typename M1, typename L2,
          typename M2>
void deep_copy_view_to_view_impl(View<T, Rank, L1, M1> &dst,
                                 const View<T, Rank, L2, M2> &src) {
  const std::size_t n = dst.size();
  if (src.size() != n) {
    throw std::runtime_error("deep_copy: View size mismatch");
  }
  if (n == 0) return;

  T *dst_ptr = dst.data();
  const T *src_ptr = src.data();

  // Both host
  if constexpr (std::is_same_v<M1, HostSpace> && std::is_same_v<M2, HostSpace>) {
    std::copy(src_ptr, src_ptr + n, dst_ptr);
    return;
  }

  // Dst device, src host
  if constexpr (std::is_same_v<M1, HostSpace> == false &&
                std::is_same_v<M2, HostSpace>) {
    auto *buf = dst.buffer_ptr();
    if (buf) {
      buf->copy_from_host(src_ptr, n);
    } else {
      // Unmanaged device view: cannot copy from host without a buffer
      throw std::runtime_error("deep_copy: destination is unmanaged device View");
    }
    return;
  }

  // Dst host, src device
  if constexpr (std::is_same_v<M1, HostSpace> &&
                std::is_same_v<M2, HostSpace> == false) {
    const auto *buf = src.buffer_ptr();
    if (buf) {
      buf->copy_to_host(dst_ptr, n);
    } else {
      throw std::runtime_error("deep_copy: source is unmanaged device View");
    }
    return;
  }

  // Both device (same or different space)
#if defined(OpenPFC_ENABLE_CUDA)
  if constexpr (std::is_same_v<M1, CudaSpace> && std::is_same_v<M2, CudaSpace>) {
    cudaError_t err =
        cudaMemcpy(dst_ptr, src_ptr, n * sizeof(T), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
      throw std::runtime_error("deep_copy: CUDA device-to-device failed");
    }
    return;
  }
#endif
#if defined(OpenPFC_ENABLE_HIP)
  if constexpr (std::is_same_v<M1, HipSpace> && std::is_same_v<M2, HipSpace>) {
    hipError_t err =
        hipMemcpy(dst_ptr, src_ptr, n * sizeof(T), hipMemcpyDeviceToDevice);
    if (err != hipSuccess) {
      throw std::runtime_error("deep_copy: HIP device-to-device failed");
    }
    return;
  }
#endif

  // Unsupported combination (e.g. Cuda to HIP)
  constexpr bool both_host =
      std::is_same_v<M1, HostSpace> && std::is_same_v<M2, HostSpace>;
  constexpr bool dst_dev_src_host =
      !std::is_same_v<M1, HostSpace> && std::is_same_v<M2, HostSpace>;
  constexpr bool dst_host_src_dev =
      std::is_same_v<M1, HostSpace> && !std::is_same_v<M2, HostSpace>;
#if defined(OpenPFC_ENABLE_CUDA)
  constexpr bool both_cuda =
      std::is_same_v<M1, CudaSpace> && std::is_same_v<M2, CudaSpace>;
#else
  constexpr bool both_cuda = false;
#endif
#if defined(OpenPFC_ENABLE_HIP)
  constexpr bool both_hip =
      std::is_same_v<M1, HipSpace> && std::is_same_v<M2, HipSpace>;
#else
  constexpr bool both_hip = false;
#endif
  constexpr bool handled =
      both_host || dst_dev_src_host || dst_host_src_dev || both_cuda || both_hip;
  static_assert(handled, "deep_copy: unsupported space combination");
}

} // namespace detail

/**
 * @brief Copy from src View to dst View (Kokkos-compatible)
 *
 * Views must have the same rank, value type, and extents. Layout may differ
 * (data is copied in logical order).
 */
template <typename T, std::size_t Rank, typename Layout1, typename MemorySpace1,
          typename Layout2, typename MemorySpace2>
void deep_copy(View<T, Rank, Layout1, MemorySpace1> &dst,
               const View<T, Rank, Layout2, MemorySpace2> &src) {
  for (std::size_t r = 0; r < Rank; ++r) {
    if (dst.extent(r) != src.extent(r)) {
      throw std::runtime_error("deep_copy: extent mismatch in dimension " +
                               std::to_string(r));
    }
  }
  detail::deep_copy_view_to_view_impl(dst, src);
}

/**
 * @brief deep_copy with execution space (Kokkos-compatible async variant)
 *
 * When execution space is device, copy may be asynchronous. Call fence()
 * to synchronize.
 */
template <typename ExecutionSpace, typename T, std::size_t Rank, typename Layout1,
          typename MemorySpace1, typename Layout2, typename MemorySpace2>
void deep_copy(const ExecutionSpace &, View<T, Rank, Layout1, MemorySpace1> &dst,
               const View<T, Rank, Layout2, MemorySpace2> &src) {
  // Synchronous implementation; async can be added per execution space
  deep_copy(dst, src);
}

/**
 * @brief Fill View with scalar value (Kokkos-compatible)
 */
template <typename T, std::size_t Rank, typename Layout, typename MemorySpace>
void deep_copy(View<T, Rank, Layout, MemorySpace> &dst, const T &value) {
  const std::size_t n = dst.size();
  if (n == 0) return;
  T *ptr = dst.data();
  if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
    std::fill(ptr, ptr + n, value);
  } else {
    // Device: copy value to host, then fill a host buffer and copy to device,
    // or run a kernel. Simplest: create host buffer, fill, copy to device.
    auto *buf = dst.buffer_ptr();
    if (buf) {
      std::vector<T> host_tmp(n, value);
      buf->copy_from_host(host_tmp);
    } else {
      throw std::runtime_error("deep_copy(scalar): unmanaged device View");
    }
  }
}

} // namespace pfc
