// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file sparse_vector_ops.hpp
 * @brief CUDA gather/scatter for SparseVector (runtime extension)
 *
 * Include this header when using gather() or scatter() with SparseVector<CudaTag,
 * T>. Kernel-only code should not include this; kernel remains backend-agnostic when
 * CUDA is disabled.
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA)

#include <openpfc/kernel/decomposition/sparse_vector_ops.hpp>
#include <openpfc/runtime/cuda/backend_tags_cuda.hpp>
#include <openpfc/runtime/cuda/sparse_vector_ops_cuda.hpp>
#include <stdexcept>

namespace pfc {
namespace core {

/** @brief Gather for SparseVector<CudaTag, double> (CUDA device). */
inline void gather(SparseVector<backend::CudaTag, double> &sparse_vector,
                   const double *source, size_t source_size) {
  if (sparse_vector.empty()) {
    return;
  }
  gather_cuda_impl(sparse_vector.size(), sparse_vector.indices().data(),
                   sparse_vector.data().data(), source, source_size);
}

/** @brief Scatter for SparseVector<CudaTag, double> (CUDA device). */
inline void scatter(const SparseVector<backend::CudaTag, double> &sparse_vector,
                    double *dest, size_t dest_size) {
  if (sparse_vector.empty()) {
    return;
  }
  scatter_cuda_impl(sparse_vector.size(), sparse_vector.indices().data(),
                    sparse_vector.data().data(), dest, dest_size);
}

} // namespace core
} // namespace pfc

#endif
