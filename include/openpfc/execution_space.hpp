// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file execution_space.hpp
 * @brief Kokkos-compatible execution space tags
 *
 * @details
 * Execution spaces define where parallel work runs. Names and semantics
 * match Kokkos so that switching to Kokkos later requires minimal changes.
 *
 * - Serial: single-threaded execution (always available)
 * - OpenMP: multi-threaded via OpenMP (optional)
 * - Cuda: GPU execution via CUDA (when OpenPFC_ENABLE_CUDA)
 * - HIP: GPU execution via HIP/ROCm (when OpenPFC_ENABLE_HIP)
 *
 * @see memory_space.hpp for where data lives
 * @see policy.hpp for RangePolicy, MDRangePolicy
 * @see parallel.hpp for parallel_for, fence
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#pragma once

namespace pfc {

/**
 * @brief Serial execution space (single-threaded)
 *
 * Work runs on the host in a single thread. Kokkos-compatible name.
 */
struct Serial {};

/**
 * @brief OpenMP execution space (multi-threaded)
 *
 * Work is distributed across threads via OpenMP. Stub for future use.
 */
struct OpenMP {};

#if defined(OpenPFC_ENABLE_CUDA)
/**
 * @brief CUDA execution space
 *
 * Work runs on GPU via CUDA kernels. Only available when OpenPFC_ENABLE_CUDA.
 */
struct Cuda {};
#endif

#if defined(OpenPFC_ENABLE_HIP)
/**
 * @brief HIP execution space
 *
 * Work runs on GPU via HIP/ROCm. Only available when OpenPFC_ENABLE_HIP.
 */
struct HIP {};
#endif

/**
 * @brief Default execution space for the current build
 *
 * Serial for CPU-only builds; Cuda or HIP when exactly one GPU backend
 * is enabled. Matches Kokkos::DefaultExecutionSpace semantics.
 */
using DefaultExecutionSpace = Serial;

} // namespace pfc
