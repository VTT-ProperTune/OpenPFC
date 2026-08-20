// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file sparse_vector_ops_gpu.hpp
 * @brief Single-source GPU SparseVector gather/scatter for CUDA and HIP (M3).
 *
 * Overloads `pfc::core::gather` / `scatter` for `SparseVector<CUDATag>` and/or
 * `SparseVector<HIPTag>`. Vendor `sparse_vector_ops.hpp` headers are thin
 * includes of this file.
 *
 * Device kernels live in `src/openpfc/runtime/gpu/sparse_vector_ops_gpu.inc`,
 * compiled from the vendor `.cu` / `.hip` translation units.
 *
 * Fail-closed OOB: any index `>=` the dense length throws `std::runtime_error`
 * with the same messages as CPU (`gather: index out of bounds` /
 * `scatter: index out of bounds`) before any device write.
 *
 * @see kernel/decomposition/sparse_vector_ops.hpp
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <cstddef>

#include <openpfc/kernel/decomposition/sparse_vector_ops.hpp>
#include <openpfc/runtime/gpu/sparse_vector_gpu.hpp>

namespace pfc::core {

#if defined(OpenPFC_ENABLE_CUDA)
/**
 * @brief Gather on device: data[i] = source[indices[i]] for i in [0, n)
 * @param n Number of entries
 * @param indices Device pointer to indices (size_t)
 * @param data Device pointer to output data (written)
 * @param source Device pointer to source array
 * @param source_size Size of source; any `indices[i] >= source_size` throws
 *        `std::runtime_error("gather: index out of bounds")` before the kernel
 *        runs (parity with CPU `pfc::core::gather`)
 */
void gather_cuda_impl(size_t n, const size_t *indices, double *data,
                      const double *source, size_t source_size);
void scatter_cuda_impl(size_t n, const size_t *indices, const double *data,
                       double *dest, size_t dest_size);

/**
 * @brief Gather for SparseVector<CUDATag, double> (CUDA device).
 * @note Fail-closed OOB parity with CPU `pfc::core::gather`.
 */
inline void gather(SparseVector<backend::CUDATag, double> &sparse_vector,
                   const double *source, size_t source_size) {
  if (sparse_vector.empty()) {
    return;
  }
  gather_cuda_impl(sparse_vector.size(), sparse_vector.indices().data(),
                   sparse_vector.data().data(), source, source_size);
}

/**
 * @brief Scatter for SparseVector<CUDATag, double> (CUDA device).
 * @note Fail-closed OOB parity with CPU `pfc::core::scatter`.
 */
inline void scatter(const SparseVector<backend::CUDATag, double> &sparse_vector,
                    double *dest, size_t dest_size) {
  if (sparse_vector.empty()) {
    return;
  }
  scatter_cuda_impl(sparse_vector.size(), sparse_vector.indices().data(),
                    sparse_vector.data().data(), dest, dest_size);
}
#endif

#if defined(OpenPFC_ENABLE_HIP)
/**
 * @brief Gather on device: data[i] = source[indices[i]] for i in [0, n)
 * @param n Number of entries
 * @param indices Device pointer to indices (size_t)
 * @param data Device pointer to output data (written)
 * @param source Device pointer to source array
 * @param source_size Size of source; any `indices[i] >= source_size` throws
 *        `std::runtime_error("gather: index out of bounds")` before the kernel
 *        runs (parity with CPU `pfc::core::gather`)
 */
void gather_hip_impl(size_t n, const size_t *indices, double *data,
                     const double *source, size_t source_size);
void scatter_hip_impl(size_t n, const size_t *indices, const double *data,
                      double *dest, size_t dest_size);

/**
 * @brief Gather for SparseVector<HIPTag, double> (HIP device).
 * @note Fail-closed OOB parity with CPU `pfc::core::gather`.
 */
inline void gather(SparseVector<backend::HIPTag, double> &sparse_vector,
                   const double *source, size_t source_size) {
  if (sparse_vector.empty()) {
    return;
  }
  gather_hip_impl(sparse_vector.size(), sparse_vector.indices().data(),
                  sparse_vector.data().data(), source, source_size);
}

/**
 * @brief Scatter for SparseVector<HIPTag, double> (HIP device).
 * @note Fail-closed OOB parity with CPU `pfc::core::scatter`.
 */
inline void scatter(const SparseVector<backend::HIPTag, double> &sparse_vector,
                    double *dest, size_t dest_size) {
  if (sparse_vector.empty()) {
    return;
  }
  scatter_hip_impl(sparse_vector.size(), sparse_vector.indices().data(),
                   sparse_vector.data().data(), dest, dest_size);
}
#endif

} // namespace pfc::core

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
