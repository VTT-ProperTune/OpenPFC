// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file memory_space.hpp
 * @brief Kokkos-compatible memory space tags (kernel: HostSpace only)
 *
 * Kernel defines HostSpace only. CudaSpace and HipSpace are in
 * runtime/gpu/memory_space_gpu.hpp (vendor headers are thin includes).
 *
 * @see runtime/gpu/memory_space_gpu.hpp for CudaSpace / HipSpace
 * @see runtime/cuda/memory_space_cuda.hpp (thin include)
 * @see runtime/hip/memory_space_hip.hpp (thin include)
 * @see kernel/execution/databuffer.hpp for buffer implementation
 */

#pragma once

#include <openpfc/kernel/execution/backend_tags.hpp>

namespace pfc {

/**
 * @brief Host (CPU) memory space
 *
 * Data is accessible from the host. Kokkos-compatible name.
 */
struct HostSpace {};

/**
 * @brief Default memory space for the current build (kernel: always HostSpace)
 */
using DefaultMemorySpace = HostSpace;

/**
 * @brief Maps a memory space tag to the corresponding backend tag
 *
 * Used internally by DataBuffer. CudaSpace and HipSpace mappings
 * are in runtime/gpu/memory_space_gpu.hpp.
 */
template <typename MemorySpace> struct memory_space_to_backend;

template <> struct memory_space_to_backend<HostSpace> {
  using type = backend::CpuTag;
};

template <typename MemorySpace>
using memory_space_to_backend_t =
    typename memory_space_to_backend<MemorySpace>::type;

} // namespace pfc
