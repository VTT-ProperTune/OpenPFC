// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file memory_space_gpu.hpp
 * @brief GPU memory space tags (M3)
 *
 * Single-source header providing both CUDA and HIP memory space tags.
 * Replaces runtime/cuda/memory_space_cuda.hpp and runtime/hip/memory_space_hip.hpp.
 *
 * @see kernel/execution/memory_space.hpp for HostSpace
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <openpfc/kernel/execution/memory_space.hpp>

#if defined(OpenPFC_ENABLE_CUDA)
#include <openpfc/runtime/cuda/backend_tags_cuda.hpp>
#elif defined(OpenPFC_ENABLE_HIP)
#include <openpfc/runtime/hip/backend_tags_hip.hpp>
#endif

namespace pfc {

#if defined(OpenPFC_ENABLE_CUDA)

struct CudaSpace {};

template <> struct memory_space_to_backend<CudaSpace> {
  using type = backend::CudaTag;
};

#elif defined(OpenPFC_ENABLE_HIP)

struct HipSpace {};

template <> struct memory_space_to_backend<HipSpace> {
  using type = backend::HipTag;
};

#endif

} // namespace pfc

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP