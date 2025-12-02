// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file backend_tags.hpp
 * @brief Backend tags for compile-time backend selection
 *
 * @details
 * This header defines empty struct tags used for template specialization
 * to select different memory and compute backends at compile-time.
 *
 * - `CpuTag`: CPU backend (always available)
 * - `CudaTag`: CUDA backend (only when OpenPFC_ENABLE_CUDA is defined)
 * - `HipTag`: HIP/ROCm backend (only when OpenPFC_ENABLE_HIP is defined)
 *
 * These tags are used with template specialization to provide backend-specific
 * implementations of data structures and operations with zero runtime overhead.
 *
 * @code
 * #include <openpfc/core/backend_tags.hpp>
 * #include <openpfc/core/databuffer.hpp>
 *
 * // CPU memory (always available)
 * pfc::core::DataBuffer<pfc::backend::CpuTag, double> cpu_buf(1000);
 *
 * #if defined(OpenPFC_ENABLE_CUDA)
 * // GPU memory (only when CUDA is enabled)
 * pfc::core::DataBuffer<pfc::backend::CudaTag, double> gpu_buf(1000);
 * #endif
 * @endcode
 *
 * @see core/databuffer.hpp for usage in memory management
 * @see core/memory_traits.hpp for backend metadata
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#pragma once

namespace pfc {
namespace backend {

/**
 * @brief CPU backend tag
 *
 * Indicates that data should be stored in host (CPU) memory and
 * operations should be performed on the CPU.
 *
 * This tag is always available, regardless of GPU support.
 */
struct CpuTag {};

#if defined(OpenPFC_ENABLE_CUDA)
/**
 * @brief CUDA backend tag
 *
 * Indicates that data should be stored in GPU memory (via CUDA)
 * and operations should be performed using CUDA kernels.
 *
 * This tag is only available when OpenPFC_ENABLE_CUDA is defined
 * and CUDA toolkit is found during configuration.
 */
struct CudaTag {};
#endif

#if defined(OpenPFC_ENABLE_HIP)
/**
 * @brief HIP/ROCm backend tag
 *
 * Indicates that data should be stored in GPU memory (via HIP/ROCm)
 * and operations should be performed using HIP kernels.
 *
 * This tag is only available when OpenPFC_ENABLE_HIP is defined
 * and HIP/ROCm is found during configuration.
 */
struct HipTag {};
#endif

} // namespace backend
} // namespace pfc
