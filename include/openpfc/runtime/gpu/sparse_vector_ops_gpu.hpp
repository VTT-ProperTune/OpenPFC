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
 * `scatter: index out of bounds`) before any device write. The check uses
 * the bound `SparseVector` records at construction, so a halo exchange does
 * not copy its index lists back to the host on every call.
 *
 * @see kernel/decomposition/sparse_vector_ops.hpp
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <concepts>
#include <cstddef>
#include <cstdint>

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
 * @param index_end One past the largest index when the caller knows it
 *        (`SparseVector::index_end()`); the check is then O(1). With
 *        `unknown_index_end` the indices are downloaded and scanned.
 */
void gather_cuda_impl(size_t n, const size_t *indices, double *data,
                      const double *source, size_t source_size,
                      size_t index_end = unknown_index_end);
void gather_cuda_impl(size_t n, const size_t *indices, float *data,
                      const float *source, size_t source_size,
                      size_t index_end = unknown_index_end);
void gather_cuda_impl(size_t n, const size_t *indices, std::uint64_t *data,
                      const std::uint64_t *source, size_t source_size,
                      size_t index_end = unknown_index_end);
void scatter_cuda_impl(size_t n, const size_t *indices, const double *data,
                       double *dest, size_t dest_size,
                       size_t index_end = unknown_index_end);

/**
 * @brief Gather for SparseVector<CUDATag, double> (CUDA device).
 * @note OOB messages match CPU; GPU validates before any write.
 */
inline void gather(SparseVector<backend::CUDATag, double> &sparse_vector,
                   const double *source, size_t source_size) {
  if (sparse_vector.empty()) {
    return;
  }
  gather_cuda_impl(sparse_vector.size(), sparse_vector.indices().data(),
                   sparse_vector.data().data(), source, source_size,
                   sparse_vector.index_end());
}

/** Compact typed selection, preserving binary32 values and full-width identities.
 * The input and index buffers must be ready for default-stream consumption;
 * callers synchronize any producer on a nondefault stream before this call.
 * Returns only after gathering completes. OOB indices leave output unchanged.
 * A nonempty selection requires nonnull source and disjoint source/output
 * storage. Empty selections do not dereference source. Bounds come from
 * `index_end()`, recorded when the indices were built on the host; only a
 * vector sized without indices is downloaded for validation.
 * Scatter remains double-only. This does not add a Float32 remapping API.
 */
template <typename T>
  requires(std::same_as<T, float> || std::same_as<T, std::uint64_t>)
inline void gather(SparseVector<backend::CUDATag, T> &sparse_vector, const T *source,
                   size_t source_size) {
  if (sparse_vector.empty()) return;
  gather_cuda_impl(sparse_vector.size(), sparse_vector.indices().data(),
                   sparse_vector.data().data(), source, source_size,
                   sparse_vector.index_end());
}

/**
 * @brief Scatter for SparseVector<CUDATag, double> (CUDA device).
 * @note OOB messages match CPU; GPU validates before any write.
 */
inline void scatter(const SparseVector<backend::CUDATag, double> &sparse_vector,
                    double *dest, size_t dest_size) {
  if (sparse_vector.empty()) {
    return;
  }
  scatter_cuda_impl(sparse_vector.size(), sparse_vector.indices().data(),
                    sparse_vector.data().data(), dest, dest_size,
                    sparse_vector.index_end());
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
 * @param index_end One past the largest index when the caller knows it
 *        (`SparseVector::index_end()`); the check is then O(1). With
 *        `unknown_index_end` the indices are downloaded and scanned.
 */
void gather_hip_impl(size_t n, const size_t *indices, double *data,
                     const double *source, size_t source_size,
                     size_t index_end = unknown_index_end);
void gather_hip_impl(size_t n, const size_t *indices, float *data,
                     const float *source, size_t source_size,
                     size_t index_end = unknown_index_end);
void gather_hip_impl(size_t n, const size_t *indices, std::uint64_t *data,
                     const std::uint64_t *source, size_t source_size,
                     size_t index_end = unknown_index_end);
void scatter_hip_impl(size_t n, const size_t *indices, const double *data,
                      double *dest, size_t dest_size,
                      size_t index_end = unknown_index_end);

/**
 * @brief Gather for SparseVector<HIPTag, double> (HIP device).
 * @note OOB messages match CPU; GPU validates before any write.
 */
inline void gather(SparseVector<backend::HIPTag, double> &sparse_vector,
                   const double *source, size_t source_size) {
  if (sparse_vector.empty()) {
    return;
  }
  gather_hip_impl(sparse_vector.size(), sparse_vector.indices().data(),
                  sparse_vector.data().data(), source, source_size,
                  sparse_vector.index_end());
}

/** Compact typed selection, preserving binary32 values and full-width identities.
 * The input and index buffers must be ready for default-stream consumption;
 * callers synchronize any producer on a nondefault stream before this call.
 * Returns only after gathering completes. OOB indices leave output unchanged.
 * A nonempty selection requires nonnull source and disjoint source/output
 * storage. Empty selections do not dereference source. Bounds come from
 * `index_end()`, recorded when the indices were built on the host; only a
 * vector sized without indices is downloaded for validation.
 * Scatter remains double-only. This does not add a Float32 remapping API.
 */
template <typename T>
  requires(std::same_as<T, float> || std::same_as<T, std::uint64_t>)
inline void gather(SparseVector<backend::HIPTag, T> &sparse_vector, const T *source,
                   size_t source_size) {
  if (sparse_vector.empty()) return;
  gather_hip_impl(sparse_vector.size(), sparse_vector.indices().data(),
                  sparse_vector.data().data(), source, source_size,
                  sparse_vector.index_end());
}

/**
 * @brief Scatter for SparseVector<HIPTag, double> (HIP device).
 * @note OOB messages match CPU; GPU validates before any write.
 */
inline void scatter(const SparseVector<backend::HIPTag, double> &sparse_vector,
                    double *dest, size_t dest_size) {
  if (sparse_vector.empty()) {
    return;
  }
  scatter_hip_impl(sparse_vector.size(), sparse_vector.indices().data(),
                   sparse_vector.data().data(), dest, dest_size,
                   sparse_vector.index_end());
}
#endif

} // namespace pfc::core

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
