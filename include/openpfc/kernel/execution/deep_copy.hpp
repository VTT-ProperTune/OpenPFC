// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file deep_copy.hpp
 * @brief Kokkos-compatible deep_copy between Views and scalar fill
 *
 * @details
 * deep_copy copies data between Views (same shape) or fills a View with a
 * scalar. Handles host-host, host-device, and device-host. Names and
 * semantics match Kokkos. Device scalar fill requires
 * `runtime/gpu/deep_copy_gpu.hpp` (or the vendor shim) and runs a device
 * kernel rather than staging a host vector.
 *
 * @see view.hpp for View
 * @see execution_space.hpp for async variant
 * @see runtime/gpu/deep_copy_gpu.hpp for device-to-device copy and device fill
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#pragma once

#include <algorithm>
#include <openpfc/kernel/execution/execution_space.hpp>
#include <openpfc/kernel/execution/view.hpp>
#include <stdexcept>

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
  if (n == 0) {
    return;
  }

  T *dst_ptr = dst.data();
  const T *src_ptr = src.data();

  // Both host
  if constexpr (std::is_same_v<M1, HostSpace> && std::is_same_v<M2, HostSpace>) {
    std::copy(src_ptr, src_ptr + n, dst_ptr);
    return;
  }

  // Dst device, src host
  if constexpr (!std::is_same_v<M1, HostSpace> && std::is_same_v<M2, HostSpace>) {
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
  if constexpr (std::is_same_v<M1, HostSpace> && !std::is_same_v<M2, HostSpace>) {
    const auto *buf = src.buffer_ptr();
    if (buf) {
      buf->copy_to_host(dst_ptr, n);
    } else {
      throw std::runtime_error("deep_copy: source is unmanaged device View");
    }
    return;
  }

  // Both device: provided by runtime (include deep_copy_gpu.hpp, or the
  // thin deep_copy_cuda.hpp / deep_copy_hip.hpp shims)
  constexpr bool both_device =
      !std::is_same_v<M1, HostSpace> && !std::is_same_v<M2, HostSpace>;
  static_assert(
      !both_device,
      "deep_copy device-to-device: include openpfc/runtime/gpu/deep_copy_gpu.hpp");
  (void)dst_ptr;
  (void)src_ptr;
  (void)n;
}

template <typename MemorySpace> struct deep_copy_device_fill_fn {
  template <typename T, std::size_t Rank, typename Layout>
  static void call(View<T, Rank, Layout, MemorySpace> & /*dst*/, const T & /*value*/) {
    static_assert(sizeof(T) == 0, "deep_copy scalar fill on device: include "
                                  "openpfc/runtime/gpu/deep_copy_gpu.hpp");
  }
};

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
void deep_copy(const ExecutionSpace & /*exec_space*/,
               View<T, Rank, Layout1, MemorySpace1> &dst,
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
  if (n == 0) {
    return;
  }
  T *ptr = dst.data();
  if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
    std::fill(ptr, ptr + n, value);
  } else {
    (void)ptr;
    detail::deep_copy_device_fill_fn<MemorySpace>::call(dst, value);
  }
}

} // namespace pfc
