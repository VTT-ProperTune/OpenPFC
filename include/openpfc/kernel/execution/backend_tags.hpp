// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file backend_tags.hpp
 * @brief Backend tags for compile-time backend selection (kernel: CPU only)
 *
 * @details
 * This header defines the CPU backend tag only. Kernel stays backend-agnostic;
 * CUDA and HIP tags are in runtime/gpu.
 *
 * - `CpuTag`: CPU backend (always available, defined here)
 * - `CudaTag` / `HipTag`: include <openpfc/runtime/gpu/backend_tags_gpu.hpp>
 *   (vendor headers `backend_tags_cuda.hpp` / `backend_tags_hip.hpp` are
 *   thin includes)
 *
 * @code
 * #include <openpfc/kernel/execution/backend_tags.hpp>
 * #include <openpfc/kernel/execution/databuffer.hpp>
 * pfc::core::DataBuffer<pfc::backend::CpuTag, double> cpu_buf(1000);
 * @endcode
 *
 * @see runtime/gpu/backend_tags_gpu.hpp for CudaTag and HipTag
 * @see runtime/cuda/backend_tags_cuda.hpp (thin include)
 * @see runtime/hip/backend_tags_hip.hpp (thin include)
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
