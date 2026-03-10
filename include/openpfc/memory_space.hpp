// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file memory_space.hpp
 * @brief Kokkos-compatible memory space tags and backend mapping
 *
 * @details
 * Memory spaces define where data is stored. Names match Kokkos.
 * Each memory space maps to an internal backend tag (CpuTag, CudaTag, HipTag)
 * used by DataBuffer and other core types.
 *
 * - HostSpace: host (CPU) memory (always available)
 * - CudaSpace: GPU memory via CUDA (when OpenPFC_ENABLE_CUDA)
 * - HipSpace: GPU memory via HIP (when OpenPFC_ENABLE_HIP)
 *
 * @see core/backend_tags.hpp for backend tags
 * @see core/databuffer.hpp for buffer implementation
 * @see execution_space.hpp for execution spaces
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#pragma once

#include <openpfc/core/backend_tags.hpp>

namespace pfc {

/**
 * @brief Host (CPU) memory space
 *
 * Data is accessible from the host. Kokkos-compatible name.
 */
struct HostSpace {};

#if defined(OpenPFC_ENABLE_CUDA)
/**
 * @brief CUDA device memory space
 *
 * Data lives in GPU memory. Only when OpenPFC_ENABLE_CUDA.
 */
struct CudaSpace {};
#endif

#if defined(OpenPFC_ENABLE_HIP)
/**
 * @brief HIP device memory space
 *
 * Data lives in GPU memory via HIP/ROCm. Only when OpenPFC_ENABLE_HIP.
 */
struct HipSpace {};
#endif

/**
 * @brief Default memory space for the current build
 *
 * HostSpace for CPU-only; matches Kokkos::DefaultExecutionSpace::memory_space.
 */
using DefaultMemorySpace = HostSpace;

/**
 * @brief Maps a memory space tag to the corresponding backend tag
 *
 * Used internally by View and deep_copy to dispatch to DataBuffer<BackendTag,T>.
 */
template <typename MemorySpace> struct memory_space_to_backend;

template <> struct memory_space_to_backend<HostSpace> {
  using type = backend::CpuTag;
};

#if defined(OpenPFC_ENABLE_CUDA)
template <> struct memory_space_to_backend<CudaSpace> {
  using type = backend::CudaTag;
};
#endif

#if defined(OpenPFC_ENABLE_HIP)
template <> struct memory_space_to_backend<HipSpace> {
  using type = backend::HipTag;
};
#endif

template <typename MemorySpace>
using memory_space_to_backend_t =
    typename memory_space_to_backend<MemorySpace>::type;

} // namespace pfc
