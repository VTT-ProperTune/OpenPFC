// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file deep_copy_hip.hpp
 * @brief HIP device-to-device deep_copy (runtime/hip)
 *
 * Include after openpfc/kernel/execution/deep_copy.hpp when copying between
 * HipSpace Views.
 */

#pragma once

#if defined(OpenPFC_ENABLE_HIP)

#include <hip/hip_runtime.h>
#include <openpfc/kernel/execution/deep_copy.hpp>
#include <openpfc/kernel/execution/view.hpp>
#include <openpfc/runtime/hip/memory_space_hip.hpp>
#include <stdexcept>

namespace pfc {
namespace detail {

template <typename T, std::size_t Rank, typename L1, typename L2>
void deep_copy_view_to_view_impl(View<T, Rank, L1, HipSpace> &dst,
                                 const View<T, Rank, L2, HipSpace> &src) {
  const std::size_t n = dst.size();
  if (src.size() != n) {
    throw std::runtime_error("deep_copy: View size mismatch");
  }
  if (n == 0) return;
  hipError_t err =
      hipMemcpy(dst.data(), src.data(), n * sizeof(T), hipMemcpyDeviceToDevice);
  if (err != hipSuccess) {
    throw std::runtime_error("deep_copy: HIP device-to-device failed");
  }
}

} // namespace detail
} // namespace pfc

#endif // OpenPFC_ENABLE_HIP
