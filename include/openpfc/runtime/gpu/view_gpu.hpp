// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file view_gpu.hpp
 * @brief Single-source GPU View execution-space mapping for CUDA and HIP (M3).
 *
 * Specializes `memory_space_execution_space` for `CudaSpace` and/or
 * `HipSpace`. Vendor headers `view_cuda.hpp` / `view_hip.hpp` are thin
 * includes of this file so existing call sites keep compiling.
 *
 * @see kernel/execution/view.hpp
 * @see runtime/gpu/memory_space_gpu.hpp
 * @see runtime/gpu/execution_space_gpu.hpp
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <openpfc/kernel/execution/view.hpp>
#include <openpfc/runtime/gpu/execution_space_gpu.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>

namespace pfc::detail {

#if defined(OpenPFC_ENABLE_CUDA)
template <> struct memory_space_execution_space<CudaSpace> {
  using type = Cuda;
};
#endif

#if defined(OpenPFC_ENABLE_HIP)
template <> struct memory_space_execution_space<HipSpace> {
  using type = HIP;
};
#endif

} // namespace pfc::detail

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
