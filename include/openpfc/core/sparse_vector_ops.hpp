// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file sparse_vector_ops.hpp
 * @brief Operations on SparseVector: gather and scatter
 *
 * @details
 * Provides gather and scatter operations for SparseVector:
 *
 * - gather: Collect values from dense array into SparseVector
 *   Direction: DenseArray → SparseVector
 *
 * - scatter: Write values from SparseVector into dense array
 *   Direction: SparseVector → DenseArray
 *
 * @code
 * // Gather: Extract values at indices from source
 * std::vector<double> source = {1.0, 2.0, 3.0, 4.0, 5.0};
 * auto sparse = sparsevector::create<double>({0, 2, 4});
 * gather(sparse, source.data(), source.size());
 * // sparse.data() now contains {1.0, 3.0, 5.0}
 *
 * // Scatter: Write values to destination at indices
 * std::vector<double> dest(5, 0.0);
 * scatter(sparse, dest.data(), dest.size());
 * // dest now contains {1.0, 0.0, 3.0, 0.0, 5.0}
 * @endcode
 *
 * @see core/sparse_vector.hpp for SparseVector definition
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#pragma once

#include <initializer_list>
#include <vector>

#include <openpfc/core/backend_tags.hpp>
#include <openpfc/core/sparse_vector.hpp>

#if defined(OpenPFC_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace pfc {
namespace core {

/**
 * @brief Gather: Collect values from dense array into SparseVector
 *
 * Reads values from dense array at sparse_vector.indices and writes to
 * sparse_vector.data.
 *
 * Direction: DenseArray → SparseVector
 *
 * @param sparse_vector SparseVector to fill
 * @param source Pointer to dense array
 * @param source_size Size of dense array
 */
template <typename BackendTag, typename T>
void gather(SparseVector<BackendTag, T> &sparse_vector, const T *source,
            size_t source_size) {
  if (sparse_vector.empty()) {
    return;
  }

  if constexpr (std::is_same_v<BackendTag, backend::CpuTag>) {
    // CPU: Direct indexed access
    const auto &indices = sparse_vector.indices();
    T *data = sparse_vector.data().data();
    for (size_t i = 0; i < sparse_vector.size(); ++i) {
      size_t idx = indices.data()[i];
      if (idx >= source_size) {
        throw std::runtime_error("gather: index out of bounds");
      }
      data[i] = source[idx];
    }
  }
#if defined(OpenPFC_ENABLE_CUDA)
  else if constexpr (std::is_same_v<BackendTag, backend::CudaTag>) {
    // CUDA: Use kernel for indexed gather
    // TODO: Implement CUDA kernel
    throw std::runtime_error("CUDA gather not yet implemented");
  }
#endif
}

/**
 * @brief Gather from std::vector (convenience overload for CPU)
 */
template <typename T>
void gather(core::SparseVector<backend::CpuTag, T> &sparse_vector,
            const std::vector<T> &source) {
  gather(sparse_vector, source.data(), source.size());
}

/**
 * @brief Gather from initializer list (convenience overload for CPU)
 */
template <typename T>
void gather(core::SparseVector<backend::CpuTag, T> &sparse_vector,
            std::initializer_list<T> source) {
  std::vector<T> vec(source);
  gather(sparse_vector, vec);
}

/**
 * @brief Scatter: Write values from SparseVector into dense array
 *
 * Writes values from sparse_vector.data into dense array at
 * sparse_vector.indices.
 *
 * Direction: SparseVector → DenseArray
 *
 * @param sparse_vector SparseVector to read from
 * @param dest Pointer to dense array
 * @param dest_size Size of dense array
 */
template <typename BackendTag, typename T>
void scatter(const SparseVector<BackendTag, T> &sparse_vector, T *dest,
             size_t dest_size) {
  if (sparse_vector.empty()) {
    return;
  }

  if constexpr (std::is_same_v<BackendTag, backend::CpuTag>) {
    // CPU: Direct indexed access
    const auto &indices = sparse_vector.indices();
    const T *data = sparse_vector.data().data();
    for (size_t i = 0; i < sparse_vector.size(); ++i) {
      size_t idx = indices.data()[i];
      if (idx >= dest_size) {
        throw std::runtime_error("scatter: index out of bounds");
      }
      dest[idx] = data[i];
    }
  }
#if defined(OpenPFC_ENABLE_CUDA)
  else if constexpr (std::is_same_v<BackendTag, backend::CudaTag>) {
    // CUDA: Use kernel for indexed scatter
    // TODO: Implement CUDA kernel
    throw std::runtime_error("CUDA scatter not yet implemented");
  }
#endif
}

/**
 * @brief Scatter to std::vector (convenience overload for CPU)
 */
template <typename T>
void scatter(const core::SparseVector<backend::CpuTag, T> &sparse_vector,
             std::vector<T> &dest) {
  scatter(sparse_vector, dest.data(), dest.size());
}

} // namespace core

// Make gather/scatter available in global namespace for backward compatibility
using core::gather;
using core::scatter;

} // namespace pfc
