// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file execution_space.hpp
 * @brief Kokkos-compatible execution space tags (kernel: Serial, OpenMP only)
 *
 * @details
 * Execution spaces define where parallel work runs. Kernel defines only
 * Serial and OpenMP; Cuda and HIP are in runtime/cuda and runtime/hip.
 *
 * - Serial, OpenMP: defined here (always available)
 * - Cuda: include <openpfc/runtime/cuda/execution_space_cuda.hpp>
 * - HIP: include <openpfc/runtime/hip/execution_space_hip.hpp>
 *
 * @see runtime/cuda/execution_space_cuda.hpp for Cuda
 * @see runtime/hip/execution_space_hip.hpp for HIP
 * @see memory_space.hpp for where data lives
 * @see parallel.hpp for parallel_for, fence
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

/**
 * @brief Default execution space for the current build (kernel: always Serial)
 */
using DefaultExecutionSpace = Serial;

} // namespace pfc
