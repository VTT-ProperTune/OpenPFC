// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file backend_tags_gpu.hpp
 * @brief Single-source GPU backend tags for CUDA and HIP (M3).
 *
 * Defines `pfc::backend::CUDATag` and `pfc::backend::HIPTag`. Vendor headers
 * `backend_tags_cuda.hpp` / `backend_tags_hip.hpp` are thin includes of this
 * file so existing call sites keep compiling. Both tags are always provided
 * so a CUDA+HIP co-enabled translation unit can name either backend (the
 * vendor headers were unguarded empty structs).
 *
 * @see kernel/execution/backend_tags.hpp
 */

#pragma once

#include <openpfc/kernel/execution/backend_tags.hpp>

namespace pfc::backend {

/**
 * @brief CUDA backend tag
 *
 * Data is stored in GPU memory via CUDA; operations use CUDA kernels.
 */
struct CUDATag {};

/**
 * @brief HIP/ROCm backend tag
 *
 * Data is stored in GPU memory via HIP/ROCm; operations use HIP kernels.
 */
struct HIPTag {};

} // namespace pfc::backend
