// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file backend_tags.hpp
 * @brief Backend tags for compile-time backend selection (kernel: CPU only)
 *
 * @details
 * This header defines the CPU backend tag only. Kernel stays backend-agnostic;
 * CUDA and HIP tags are in runtime/cuda and runtime/hip.
 *
 * - `CpuTag`: CPU backend (always available, defined here)
 * - `CudaTag`: include <openpfc/runtime/cuda/backend_tags_cuda.hpp>
 * - `HipTag`: include <openpfc/runtime/hip/backend_tags_hip.hpp>
 *
 * @code
 * #include <openpfc/kernel/execution/backend_tags.hpp>
 * #include <openpfc/kernel/execution/databuffer.hpp>
 * pfc::core::DataBuffer<pfc::backend::CpuTag, double> cpu_buf(1000);
 * @endcode
 *
 * @see runtime/cuda/backend_tags_cuda.hpp for CudaTag
 * @see runtime/hip/backend_tags_hip.hpp for HipTag
 * @see kernel/execution/databuffer.hpp for usage in memory management
 */

#pragma once

namespace pfc::backend {

/**
 * @brief CPU backend tag
 *
 * Indicates that data should be stored in host (CPU) memory and
 * operations should be performed on the CPU. Always available.
 */
struct CpuTag {};

} // namespace pfc::backend
