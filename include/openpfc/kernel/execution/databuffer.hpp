// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file databuffer.hpp
 * @brief Backend-agnostic memory buffer with tag-based dispatch
 *
 * @details
 * DataBuffer provides a unified interface for memory management across different
 * backends (CPU, CUDA, HIP) using template specialization. Each backend is
 * conditionally compiled, ensuring that CPU-only builds work without GPU
 * dependencies.
 *
 * Key features:
 * - Backend-agnostic interface (same API for all backends)
 * - Zero runtime overhead (compile-time dispatch)
 * - RAII memory management
 * - CPU-GPU transfer methods
 * - Move semantics supported
 * - Copy semantics disabled for GPU backends
 *
 * @code
 * // CPU memory (always available)
 * pfc::DataBuffer<pfc::backend::CpuTag, double> cpu_buf(1000);
 * cpu_buf[0] = 1.0;  // Works - CPU has operator[]
 *
 * // GPU memory: include openpfc/runtime/cuda/databuffer_cuda.hpp
 * // (and backend_tags_cuda.hpp as needed)
 * pfc::core::DataBuffer<pfc::backend::CudaTag, double> gpu_buf(1000);
 * std::vector<double> host_data(1000, 1.0);
 * gpu_buf.copy_from_host(host_data);
 * @endcode
 *
 * @see kernel/execution/backend_tags.hpp for backend tag definitions
 * @see kernel/execution/memory_traits.hpp for backend metadata
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/execution/backend_tags.hpp>

namespace pfc::core {

template <typename T> constexpr bool dependent_false_databuffer = false;

/**
 * @brief Backend-agnostic memory buffer
 *
 * Template class that provides unified interface for memory management
 * across different backends. Kernel defines CpuTag only; include
 * runtime/cuda or runtime/hip headers for GPU backends.
 *
 * @tparam BackendTag Backend tag (CpuTag in kernel; CudaTag/HipTag in runtime)
 * @tparam T Element type (must be trivially copyable for GPU backends)
 */
template <typename BackendTag, typename T> struct DataBuffer;

/**
 * @brief CPU specialization of DataBuffer
 *
 * Uses std::vector for host memory management.
 * Provides operator[] for direct element access.
 */
template <typename T> struct DataBuffer<backend::CpuTag, T> {
private:
  std::vector<T> m_data;

public:
  /**
   * @brief Constructs a CPU buffer with the given size
   * @param size Number of elements
   */
  explicit DataBuffer(size_t size) : m_data(size) {}

  /**
   * @brief Default constructor (empty buffer)
   */
  DataBuffer() = default;

  /**
   * @brief Copy constructor
   */
  DataBuffer(const DataBuffer &) = default;

  /**
   * @brief Copy assignment
   */
  DataBuffer &operator=(const DataBuffer &) = default;

  /**
   * @brief Move constructor
   */
  DataBuffer(DataBuffer &&) noexcept = default;

  /**
   * @brief Move assignment
   */
  DataBuffer &operator=(DataBuffer &&) noexcept = default;

  /**
   * @brief Destructor
   */
  ~DataBuffer() = default;

  /**
   * @brief Returns pointer to underlying data
   * @return Pointer to data (host memory)
   */
  T *data() { return m_data.data(); }

  /**
   * @brief Returns const pointer to underlying data
   * @return Const pointer to data (host memory)
   */
  const T *data() const { return m_data.data(); }

  /**
   * @brief Returns the number of elements
   * @return Size of buffer
   */
  size_t size() const { return m_data.size(); }

  /**
   * @brief Returns true if buffer is empty
   * @return True if size() == 0
   */
  bool empty() const { return m_data.empty(); }

  /**
   * @brief Element access operator (CPU only)
   * @param i Element index
   * @return Reference to element at index i
   */
  T &operator[](size_t i) { return m_data[i]; }

  /**
   * @brief Const element access operator (CPU only)
   * @param i Element index
   * @return Const reference to element at index i
   */
  const T &operator[](size_t i) const { return m_data[i]; }

  /**
   * @brief Get underlying std::vector reference (for Model compatibility)
   *
   * This method allows DataBuffer<CpuTag, T> to be used with Model's
   * add_real_field() and add_complex_field() methods which expect
   * std::vector<T>& references.
   *
   * @return Reference to underlying std::vector
   *
   * @note Only available for CpuTag specialization
   * @note For GPU backends, this method is not available
   */
  std::vector<T> &as_vector() { return m_data; }

  /**
   * @brief Get underlying std::vector const reference (for Model compatibility)
   * @return Const reference to underlying std::vector
   */
  const std::vector<T> &as_vector() const { return m_data; }

  /**
   * @brief Copy data from host vector
   * @param src Source vector (must have same size)
   * @throws std::runtime_error if sizes don't match
   */
  void copy_from_host(const std::vector<T> &src) {
    if (src.size() != m_data.size()) {
      throw std::runtime_error("Size mismatch in copy_from_host: expected " +
                               std::to_string(m_data.size()) + ", got " +
                               std::to_string(src.size()));
    }
    std::copy(src.begin(), src.end(), m_data.begin());
  }

  /**
   * @brief Copy data from host pointer
   * @param ptr Source pointer (must have at least size() elements)
   * @param n Number of elements (must equal size())
   */
  void copy_from_host(const T *ptr, size_t n) {
    if (n != m_data.size()) {
      throw std::runtime_error("Size mismatch in copy_from_host: expected " +
                               std::to_string(m_data.size()) + ", got " +
                               std::to_string(n));
    }
    std::copy(ptr, ptr + n, m_data.begin());
  }

  /**
   * @brief Copy data from a host contiguous range
   * @param src Source span (size() must equal this buffer's size())
   */
  void copy_from_host(std::span<const T> src) {
    copy_from_host(src.data(), src.size());
  }

  /**
   * @brief Copy data to host pointer
   * @param ptr Destination pointer (must have space for size() elements)
   * @param n Number of elements (must equal size())
   */
  void copy_to_host(T *ptr, size_t n) const {
    if (n != m_data.size()) {
      throw std::runtime_error("Size mismatch in copy_to_host: expected " +
                               std::to_string(m_data.size()) + ", got " +
                               std::to_string(n));
    }
    std::copy(m_data.begin(), m_data.end(), ptr);
  }

  /**
   * @brief Copy data to host vector
   * @return Vector containing buffer data
   */
  std::vector<T> to_host() const { return m_data; }

  /**
   * @brief Resize the buffer
   * @param new_size New size
   */
  void resize(size_t new_size) { m_data.resize(new_size); }
};

/**
 * @brief Unsupported backend: include openpfc/runtime/cuda or runtime/hip for GPU
 *
 * Kernel only provides DataBuffer<CpuTag,T>. For CudaTag or HipTag, include
 * the corresponding runtime header.
 */
template <typename BackendTag, typename T> struct DataBuffer {
  static_assert(dependent_false_databuffer<BackendTag>,
                "DataBuffer: include openpfc/runtime/cuda or openpfc/runtime/hip "
                "for GPU backends");
};

} // namespace pfc::core
