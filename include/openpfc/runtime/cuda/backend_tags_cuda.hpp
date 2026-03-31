// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file backend_tags_cuda.hpp
 * @brief CUDA backend tag (runtime/cuda only)
 *
 * Include this header when using CudaTag. Kernel defines only CpuTag; CUDA and
 * HIP tags live in runtime so kernel stays backend-agnostic.
 *
 * @see kernel/execution/backend_tags.hpp for CpuTag
 * @see runtime/hip/backend_tags_hip.hpp for HipTag
 */

#pragma once

#include <openpfc/kernel/execution/backend_tags.hpp>

namespace pfc::backend {

/**
 * @brief CUDA backend tag
 *
 * Indicates that data should be stored in GPU memory (via CUDA) and operations
 * should be performed using CUDA kernels. Only available when building with
 * OpenPFC_ENABLE_CUDA; define this header only in runtime/cuda.
 */
struct CudaTag {};

} // namespace pfc::backend
