// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file kernels_simple.hpp
 * @brief Simple GPU kernel operations for element-wise computations
 *
 * @details
 * This header provides simple GPU kernel operations that work with GPUVector.
 * These are basic building blocks for more complex GPU computations.
 *
 * All functions are only available when CUDA is enabled at compile time.
 * On systems without CUDA, these functions will not be compiled.
 *
 * @code
 * #ifdef OpenPFC_ENABLE_CUDA
 *     pfc::gpu::GPUVector<double> vec(1000);
 *     // ... initialize vec ...
 *     pfc::gpu::add_scalar(vec, 10.0);  // Add 10.0 to each element
 * #endif
 * @endcode
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef PFC_GPU_KERNELS_SIMPLE_HPP
#define PFC_GPU_KERNELS_SIMPLE_HPP

#include <openpfc/gpu/gpu_vector.hpp>

namespace pfc {
namespace gpu {

/**
 * @brief Add a scalar value to each element of a GPU vector
 *
 * Performs element-wise addition: `vec[i] = vec[i] + value` for all i.
 *
 * @param vec GPU vector to modify (in-place)
 * @param value Scalar value to add to each element
 *
 * @throws std::runtime_error if CUDA kernel launch fails
 *
 * @note Only available when OpenPFC_ENABLE_CUDA is defined.
 *       On systems without CUDA, this function will not be compiled.
 */
void add_scalar(GPUVector<double> &vec, double value);

/**
 * @brief Multiply each element of a GPU vector by a scalar
 *
 * Performs element-wise multiplication: `vec[i] = vec[i] * value` for all i.
 *
 * @param vec GPU vector to modify (in-place)
 * @param value Scalar value to multiply each element by
 *
 * @throws std::runtime_error if CUDA kernel launch fails
 *
 * @note Only available when OpenPFC_ENABLE_CUDA is defined.
 */
void multiply_scalar(GPUVector<double> &vec, double value);

} // namespace gpu
} // namespace pfc

#endif
