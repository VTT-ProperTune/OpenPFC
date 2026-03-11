// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file sparse_vector_ops_cuda.hpp
 * @brief CUDA implementation of gather/scatter for SparseVector (device pointers)
 *
 * Declarations for gather_cuda_impl and scatter_cuda_impl. Only included and
 * linked when OpenPFC_ENABLE_CUDA is defined. Use
 * openpfc/runtime/cuda/sparse_vector_ops.hpp for gather/scatter overloads for
 * SparseVector<CudaTag, T>.
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA)

#include <cstddef>

namespace pfc {
namespace core {

/**
 * @brief Gather on device: data[i] = source[indices[i]] for i in [0, n)
 * @param n Number of entries
 * @param indices Device pointer to indices (size_t)
 * @param data Device pointer to output data (written)
 * @param source Device pointer to source array
 * @param source_size Size of source (for bounds check)
 */
void gather_cuda_impl(size_t n, const size_t *indices, double *data,
                      const double *source, size_t source_size);

/**
 * @brief Scatter on device: dest[indices[i]] = data[i] for i in [0, n)
 * @param n Number of entries
 * @param indices Device pointer to indices (size_t)
 * @param data Device pointer to input data (read)
 * @param dest Device pointer to destination array
 * @param dest_size Size of dest (for bounds check)
 */
void scatter_cuda_impl(size_t n, const size_t *indices, const double *data,
                       double *dest, size_t dest_size);

} // namespace core
} // namespace pfc

#endif
