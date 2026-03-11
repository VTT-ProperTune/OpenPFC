// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file backend_tags_hip.hpp
 * @brief HIP backend tag (runtime/hip only)
 *
 * Include this header when using HipTag. Kernel defines only CpuTag; CUDA and
 * HIP tags live in runtime so kernel stays backend-agnostic.
 *
 * @see kernel/execution/backend_tags.hpp for CpuTag
 * @see runtime/cuda/backend_tags_cuda.hpp for CudaTag
 */

#pragma once

#include <openpfc/kernel/execution/backend_tags.hpp>

namespace pfc {
namespace backend {

/**
 * @brief HIP/ROCm backend tag
 *
 * Indicates that data should be stored in GPU memory (via HIP/ROCm) and
 * operations should be performed using HIP kernels. Only available when
 * building with OpenPFC_ENABLE_HIP; define this header only in runtime/hip.
 */
struct HipTag {};

} // namespace backend
} // namespace pfc
