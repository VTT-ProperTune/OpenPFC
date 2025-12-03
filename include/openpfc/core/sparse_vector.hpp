// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file sparse_vector.hpp
 * @brief Sparse vector for halo exchange and indexed data views
 *
 * @details
 * SparseVector represents a sparse view into selected entries of a dense array.
 * It contains:
 * - Sorted indices (exchanged once during setup)
 * - Values at those indices (exchanged every step)
 *
 * Key optimization: Indices are sorted for optimal contiguous memory access.
 *
 * Used for:
 * - Halo exchange in distributed memory
 * - Arbitrary subregion extraction
 * - Mask-based data selection
 *
 * @code
 * // Create sparse vector with indices
 * std::vector<size_t> indices = {2, 4, 6};
 * auto sparse = pfc::core::SparseVector<pfc::backend::CpuTag, double>(indices);
 *
 * // Gather values from source array
 * std::vector<double> source = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
 * gather(sparse, source.data(), source.size());
 *
 * // Scatter values to destination
 * std::vector<double> dest(7, 0.0);
 * scatter(sparse, dest.data(), dest.size());
 * @endcode
 *
 * @see core/databuffer.hpp for underlying memory management
 * @see core/exchange.hpp for MPI communication
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include <openpfc/core/backend_tags.hpp>
#include <openpfc/core/databuffer.hpp>

namespace pfc {
namespace core {

// Alias for backward compatibility
using HostTag = backend::CpuTag;

/**
 * @brief Sparse vector for indexed data views
 *
 * Represents a sparse view into selected entries of a dense array.
 * Indices are automatically sorted for optimal memory access.
 *
 * @tparam BackendTag Backend tag (CpuTag, CudaTag, HipTag)
 * @tparam T Element type (must be trivially copyable for GPU backends)
 */
template <typename BackendTag, typename T> class SparseVector {
private:
  DataBuffer<BackendTag, size_t> m_indices; // Sorted indices
  DataBuffer<BackendTag, T> m_data;         // Values at those indices
  size_t m_size;
  bool m_indices_sorted;

  /**
   * @brief Copy sorted indices to device
   */
  void copy_indices_to_device(const std::vector<size_t> &sorted_indices) {
    if constexpr (std::is_same_v<BackendTag, backend::CpuTag>) {
      // CPU: Direct copy
      std::copy(sorted_indices.begin(), sorted_indices.end(), m_indices.data());
    }
#if defined(OpenPFC_ENABLE_CUDA)
    else if constexpr (std::is_same_v<BackendTag, backend::CudaTag>) {
      // CUDA: Use cudaMemcpy
      cudaError_t err = cudaMemcpy(m_indices.data(), sorted_indices.data(),
                                   m_size * sizeof(size_t), cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        throw std::runtime_error("CUDA copy failed: " +
                                 std::string(cudaGetErrorString(err)));
      }
    }
#endif
  }

public:
  /**
   * @brief Construct empty SparseVector
   * @param size Number of entries
   */
  explicit SparseVector(size_t size)
      : m_indices(size), m_data(size), m_size(size), m_indices_sorted(false) {}

  /**
   * @brief Construct SparseVector with given indices
   * @param indices Linear indices into dense array (will be sorted)
   */
  explicit SparseVector(const std::vector<size_t> &indices)
      : m_size(indices.size()), m_indices_sorted(false) {
    if (m_size == 0) {
      m_indices = DataBuffer<BackendTag, size_t>(0);
      m_data = DataBuffer<BackendTag, T>(0);
      m_indices_sorted = true;
      return;
    }

    // Sort indices for optimal memory access
    std::vector<size_t> sorted_indices = indices;
    std::sort(sorted_indices.begin(), sorted_indices.end());
    m_indices_sorted = true;

    // Allocate buffers
    m_indices = DataBuffer<BackendTag, size_t>(m_size);
    m_data = DataBuffer<BackendTag, T>(m_size);

    // Copy sorted indices to device
    copy_indices_to_device(sorted_indices);
  }

  /**
   * @brief Construct SparseVector with indices and initial data
   * @param indices Linear indices (will be sorted)
   * @param data Initial values
   */
  SparseVector(const std::vector<size_t> &indices, const std::vector<T> &data)
      : m_size(indices.size()), m_indices_sorted(false) {
    if (indices.size() != data.size()) {
      throw std::runtime_error("SparseVector: indices and data size mismatch");
    }

    if (m_size == 0) {
      m_indices = DataBuffer<BackendTag, size_t>(0);
      m_data = DataBuffer<BackendTag, T>(0);
      m_indices_sorted = true;
      return;
    }

    // Sort indices and reorder data accordingly
    std::vector<std::pair<size_t, T>> pairs;
    pairs.reserve(m_size);
    for (size_t i = 0; i < m_size; ++i) {
      pairs.emplace_back(indices[i], data[i]);
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    std::vector<size_t> sorted_indices;
    std::vector<T> sorted_data;
    sorted_indices.reserve(m_size);
    sorted_data.reserve(m_size);
    for (const auto &[idx, val] : pairs) {
      sorted_indices.push_back(idx);
      sorted_data.push_back(val);
    }
    m_indices_sorted = true;

    // Allocate buffers
    m_indices = DataBuffer<BackendTag, size_t>(m_size);
    m_data = DataBuffer<BackendTag, T>(m_size);

    // Copy to device
    copy_indices_to_device(sorted_indices);
    copy_data_to_device(sorted_data);
  }

  /**
   * @brief Get number of entries
   */
  size_t size() const { return m_size; }

  /**
   * @brief Check if empty
   */
  bool empty() const { return m_size == 0; }

  /**
   * @brief Get indices buffer (read-only)
   */
  const DataBuffer<BackendTag, size_t> &indices() const { return m_indices; }

  /**
   * @brief Get data buffer (read-write)
   */
  DataBuffer<BackendTag, T> &data() { return m_data; }
  const DataBuffer<BackendTag, T> &data() const { return m_data; }

  /**
   * @brief Check if indices are sorted
   */
  bool is_sorted() const { return m_indices_sorted; }

private:
  /**
   * @brief Copy data to device
   */
  void copy_data_to_device(const std::vector<T> &host_data) {
    if constexpr (std::is_same_v<BackendTag, backend::CpuTag>) {
      // CPU: Direct copy
      std::copy(host_data.begin(), host_data.end(), m_data.data());
    }
#if defined(OpenPFC_ENABLE_CUDA)
    else if constexpr (std::is_same_v<BackendTag, backend::CudaTag>) {
      // CUDA: Use cudaMemcpy
      cudaError_t err = cudaMemcpy(m_data.data(), host_data.data(),
                                   m_size * sizeof(T), cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        throw std::runtime_error("CUDA copy failed: " +
                                 std::string(cudaGetErrorString(err)));
      }
    }
#endif
  }
};

/**
 * @brief Check if SparseVector is on host
 */
template <typename BackendTag, typename T>
bool on_host(const SparseVector<BackendTag, T> &vec) {
  (void)vec; // Unused parameter
  return std::is_same_v<BackendTag, backend::CpuTag>;
}

} // namespace core

// Convenience namespace for backward compatibility with tests
namespace sparsevector {

// Alias for backward compatibility
using HostTag = backend::CpuTag;

/**
 * @brief Create empty SparseVector
 */
template <typename T, typename BackendTag = backend::CpuTag>
core::SparseVector<BackendTag, T> create(size_t size) {
  return core::SparseVector<BackendTag, T>(size);
}

/**
 * @brief Create SparseVector with indices (from initializer list)
 */
template <typename T, typename BackendTag = backend::CpuTag>
core::SparseVector<BackendTag, T> create(std::initializer_list<size_t> indices) {
  return core::SparseVector<BackendTag, T>(std::vector<size_t>(indices));
}

/**
 * @brief Create SparseVector with indices (from vector)
 */
template <typename T, typename BackendTag = backend::CpuTag>
core::SparseVector<BackendTag, T> create(const std::vector<size_t> &indices) {
  return core::SparseVector<BackendTag, T>(indices);
}

/**
 * @brief Create SparseVector with indices and data
 */
template <typename T, typename BackendTag = backend::CpuTag>
core::SparseVector<BackendTag, T> create(const std::vector<size_t> &indices,
                                         const std::vector<T> &data) {
  return core::SparseVector<BackendTag, T>(indices, data);
}

/**
 * @brief Get size of SparseVector
 */
template <typename BackendTag, typename T>
size_t get_size(const core::SparseVector<BackendTag, T> &vec) {
  return vec.size();
}

/**
 * @brief Set indices (CPU only - for testing)
 *
 * Note: This recreates the SparseVector internally. For production code,
 * create a new SparseVector instead.
 */
template <typename BackendTag, typename T>
void set_index(core::SparseVector<BackendTag, T> &vec,
               const std::vector<size_t> &indices) {
  static_assert(std::is_same_v<BackendTag, backend::CpuTag>,
                "set_index only available for CPU");
  // Recreate vector with new indices
  // This is a workaround for testing - in production, create new SparseVector
  core::SparseVector<BackendTag, T> new_vec(indices);
  vec = std::move(new_vec);
}

/**
 * @brief Set data (CPU only - for testing)
 */
template <typename BackendTag, typename T>
void set_data(core::SparseVector<BackendTag, T> &vec, const std::vector<T> &data) {
  static_assert(std::is_same_v<BackendTag, backend::CpuTag>,
                "set_data only available for CPU");
  if (data.size() != vec.size()) {
    throw std::runtime_error("set_data: size mismatch");
  }
  // Direct copy to data buffer (mutable access)
  T *data_ptr = vec.data().data();
  std::copy(data.begin(), data.end(), data_ptr);
}

/**
 * @brief Get indices (CPU only - for testing)
 */
template <typename BackendTag, typename T>
std::vector<size_t> get_index(const core::SparseVector<BackendTag, T> &vec) {
  static_assert(std::is_same_v<BackendTag, backend::CpuTag>,
                "get_index only available for CPU");
  std::vector<size_t> result(vec.size());
  std::copy(vec.indices().data(), vec.indices().data() + vec.size(), result.begin());
  return result;
}

/**
 * @brief Get data (CPU only - for testing)
 */
template <typename BackendTag, typename T>
std::vector<T> get_data(const core::SparseVector<BackendTag, T> &vec) {
  static_assert(std::is_same_v<BackendTag, backend::CpuTag>,
                "get_data only available for CPU");
  std::vector<T> result(vec.size());
  std::copy(vec.data().data(), vec.data().data() + vec.size(), result.begin());
  return result;
}

/**
 * @brief Get single index (CPU only)
 */
template <typename BackendTag, typename T>
size_t get_index(const core::SparseVector<BackendTag, T> &vec, size_t i) {
  static_assert(std::is_same_v<BackendTag, backend::CpuTag>,
                "get_index only available for CPU");
  return vec.indices().data()[i];
}

/**
 * @brief Get single data value (CPU only)
 */
template <typename BackendTag, typename T>
T get_data(const core::SparseVector<BackendTag, T> &vec, size_t i) {
  static_assert(std::is_same_v<BackendTag, backend::CpuTag>,
                "get_data only available for CPU");
  return vec.data().data()[i];
}

/**
 * @brief Get entry as pair (CPU only)
 */
template <typename BackendTag, typename T>
std::pair<size_t, T> get_entry(const core::SparseVector<BackendTag, T> &vec,
                               size_t i) {
  static_assert(std::is_same_v<BackendTag, backend::CpuTag>,
                "get_entry only available for CPU");
  return {vec.indices().data()[i], vec.data().data()[i]};
}

} // namespace sparsevector

} // namespace pfc
