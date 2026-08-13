// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file memory_traits_gpu.hpp
 * @brief Single-source GPU `backend_traits` for CUDA and HIP (M3).
 *
 * Specializes `pfc::core::backend_traits` for `CudaTag` and/or `HipTag`.
 * Vendor headers `memory_traits_cuda.hpp` / `memory_traits_hip.hpp` are thin
 * includes of this file so existing call sites keep compiling.
 *
 * @see kernel/execution/memory_traits.hpp
 * @see runtime/gpu/backend_tags_gpu.hpp
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <openpfc/kernel/execution/memory_traits.hpp>
#include <openpfc/runtime/gpu/backend_tags_gpu.hpp>

namespace pfc::core {

namespace detail {

struct GpuBackendTraits {
  static constexpr bool has_host_access = false;
  static constexpr bool has_device_access = true;
  static constexpr bool requires_transfer = true;
};

} // namespace detail

#if defined(OpenPFC_ENABLE_CUDA)
template <> struct backend_traits<backend::CudaTag> : detail::GpuBackendTraits {};
#endif

#if defined(OpenPFC_ENABLE_HIP)
template <> struct backend_traits<backend::HipTag> : detail::GpuBackendTraits {};
#endif

} // namespace pfc::core

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
