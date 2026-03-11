// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file execution_space_cuda.hpp
 * @brief CUDA execution space tag (runtime/cuda only)
 *
 * @see kernel/execution/execution_space.hpp for Serial, OpenMP
 * @see runtime/hip/execution_space_hip.hpp for HIP
 */

#pragma once

#include <openpfc/kernel/execution/execution_space.hpp>

namespace pfc {

/**
 * @brief CUDA execution space
 *
 * Work runs on GPU via CUDA kernels. Include this header when using Cuda.
 */
struct Cuda {};

} // namespace pfc
