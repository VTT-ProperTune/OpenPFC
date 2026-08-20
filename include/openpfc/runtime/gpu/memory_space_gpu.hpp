// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file memory_space_gpu.hpp
 * @brief Single-source GPU memory space tags for CUDA and HIP (M3).
 *
 * Defines `CUDASpace` and/or `HIPSpace` plus `memory_space_to_backend`
 * specializations depending on the enabled backends. Vendor headers
 * `memory_space_cuda.hpp` / `memory_space_hip.hpp` are thin includes of this
 * file so existing call sites keep compiling.
 *
 * Per-tag types stay independent so a CUDA+HIP co-enabled translation unit
 * can name both spaces.
 *
 * @see kernel/execution/memory_space.hpp
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/runtime/gpu/backend_tags_gpu.hpp>

namespace pfc {

#if defined(OpenPFC_ENABLE_CUDA)
struct CUDASpace {};

template <> struct memory_space_to_backend<CUDASpace> {
  using type = backend::CUDATag;
};
#endif

#if defined(OpenPFC_ENABLE_HIP)
struct HIPSpace {};

template <> struct memory_space_to_backend<HIPSpace> {
  using type = backend::HIPTag;
};
#endif

} // namespace pfc

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
