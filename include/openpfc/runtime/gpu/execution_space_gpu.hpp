// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file execution_space_gpu.hpp
 * @brief Single-source GPU execution space tags for CUDA and HIP (M3).
 *
 * Defines `pfc::Cuda` and `pfc::HIP`. Vendor headers `execution_space_cuda.hpp`
 * / `execution_space_hip.hpp` are thin includes of this file so existing call
 * sites keep compiling. Both tags are always provided so a CUDA+HIP
 * co-enabled translation unit can name either space (the vendor headers were
 * unguarded empty structs).
 *
 * @see kernel/execution/execution_space.hpp
 */

#pragma once

#include <openpfc/kernel/execution/execution_space.hpp>

namespace pfc {

/**
 * @brief CUDA execution space
 *
 * Work runs on GPU via CUDA kernels.
 */
struct Cuda {};

/**
 * @brief HIP execution space
 *
 * Work runs on GPU via HIP/ROCm.
 */
struct HIP {};

} // namespace pfc
