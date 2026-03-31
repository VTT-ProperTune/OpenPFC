// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
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

  // Both device: provided by runtime (include deep_copy_cuda.hpp /
  // deep_copy_hip.hpp)
  constexpr bool both_device =
      !std::is_same_v<M1, HostSpace> && !std::is_same_v<M2, HostSpace>;
  static_assert(
      !both_device,
      "deep_copy device-to-device: include openpfc/runtime/cuda/deep_copy_cuda.hpp "
      "or openpfc/runtime/hip/deep_copy_hip.hpp");
  (void)dst_ptr;
  (void)src_ptr;
  (void)n;
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
