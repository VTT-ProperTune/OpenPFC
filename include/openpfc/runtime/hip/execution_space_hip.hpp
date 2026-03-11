// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file execution_space_hip.hpp
 * @brief HIP execution space tag (runtime/hip only)
 *
 * @see kernel/execution/execution_space.hpp for Serial, OpenMP
 * @see runtime/cuda/execution_space_cuda.hpp for Cuda
 */

#pragma once

#include <openpfc/kernel/execution/execution_space.hpp>

namespace pfc {

/**
 * @brief HIP execution space
 *
 * Work runs on GPU via HIP/ROCm. Include this header when using HIP.
 */
struct HIP {};

} // namespace pfc
