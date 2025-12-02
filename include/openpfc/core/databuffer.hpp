// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
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
 * pfc::core::DataBuffer<pfc::backend::CpuTag, double> cpu_buf(1000);
 * cpu_buf[0] = 1.0;  // Works - CPU has operator[]
 *
 * #if defined(OpenPFC_ENABLE_CUDA)
 * // GPU memory (only when CUDA enabled)
 * pfc::core::DataBuffer<pfc::backend::CudaTag, double> gpu_buf(1000);
 * // gpu_buf[0] = 1.0;  // ERROR - can't dereference device pointer!
 * std::vector<double> host_data(1000, 1.0);
 * gpu_buf.copy_from_host(host_data);
 * #endif
 * @endcode
 *
 * @see core/backend_tags.hpp for backend tag definitions
 * @see core/memory_traits.hpp for backend metadata
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <openpfc/core/backend_tags.hpp>

// CUDA headers (only when CUDA is enabled)
#if defined(OpenPFC_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace pfc {
namespace core {

/**
 * @brief Backend-agnostic memory buffer
 *
 * Template class that provides unified interface for memory management
 * across different backends. Specialized for each backend tag.
 *
 * @tparam BackendTag Backend tag (CpuTag, CudaTag, HipTag)
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

#if defined(OpenPFC_ENABLE_CUDA)
/**
 * @brief CUDA specialization of DataBuffer
 *
 * Uses CUDA memory allocation (cudaMalloc/cudaFree).
 * Does not provide operator[] (can't dereference device pointer on host).
 */
template <typename T> struct DataBuffer<backend::CudaTag, T> {
private:
  T *m_device_ptr = nullptr;
  size_t m_size = 0;

public:
  /**
   * @brief Constructs a CUDA buffer with the given size
   * @param size Number of elements
   * @throws std::runtime_error if CUDA allocation fails
   */
  explicit DataBuffer(size_t size) : m_size(size) {
    if (size > 0) {
      cudaError_t err = cudaMalloc(&m_device_ptr, size * sizeof(T));
      if (err != cudaSuccess) {
        throw std::runtime_error("CUDA allocation failed: " +
                                 std::string(cudaGetErrorString(err)));
      }
    }
  }

  /**
   * @brief Default constructor (empty buffer)
   */
  DataBuffer() = default;

  /**
   * @brief Destructor (frees CUDA memory)
   */
  ~DataBuffer() {
    if (m_device_ptr != nullptr) {
      cudaFree(m_device_ptr);
    }
  }

  /**
   * @brief Copy constructor is deleted (would require deep copy)
   */
  DataBuffer(const DataBuffer &) = delete;

  /**
   * @brief Copy assignment is deleted (would require deep copy)
   */
  DataBuffer &operator=(const DataBuffer &) = delete;

  /**
   * @brief Move constructor
   */
  DataBuffer(DataBuffer &&other) noexcept
      : m_device_ptr(other.m_device_ptr), m_size(other.m_size) {
    other.m_device_ptr = nullptr;
    other.m_size = 0;
  }

  /**
   * @brief Move assignment
   */
  DataBuffer &operator=(DataBuffer &&other) noexcept {
    if (this != &other) {
      if (m_device_ptr != nullptr) {
        cudaFree(m_device_ptr);
      }
      m_device_ptr = other.m_device_ptr;
      m_size = other.m_size;
      other.m_device_ptr = nullptr;
      other.m_size = 0;
    }
    return *this;
  }

  /**
   * @brief Returns pointer to underlying data (device memory)
   * @return Pointer to data (device memory)
   */
  T *data() { return m_device_ptr; }

  /**
   * @brief Returns const pointer to underlying data (device memory)
   * @return Const pointer to data (device memory)
   */
  const T *data() const { return m_device_ptr; }

  /**
   * @brief Returns the number of elements
   * @return Size of buffer
   */
  size_t size() const { return m_size; }

  /**
   * @brief Returns true if buffer is empty
   * @return True if size() == 0
   */
  bool empty() const { return m_size == 0; }

  // Note: No operator[] - can't dereference device pointer on host!

  /**
   * @brief Copy data from host vector to device
   * @param src Source vector (must have same size)
   * @throws std::runtime_error if sizes don't match or copy fails
   */
  void copy_from_host(const std::vector<T> &src) {
    if (src.size() != m_size) {
      throw std::runtime_error("Size mismatch in copy_from_host: expected " +
                               std::to_string(m_size) + ", got " +
                               std::to_string(src.size()));
    }
    if (m_size > 0) {
      cudaError_t err = cudaMemcpy(m_device_ptr, src.data(), m_size * sizeof(T),
                                   cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        throw std::runtime_error("CUDA copy failed: " +
                                 std::string(cudaGetErrorString(err)));
      }
    }
  }

  /**
   * @brief Copy data from device to host vector
   * @return Vector containing buffer data
   * @throws std::runtime_error if copy fails
   */
  std::vector<T> to_host() const {
    std::vector<T> result(m_size);
    if (m_size > 0) {
      cudaError_t err = cudaMemcpy(result.data(), m_device_ptr, m_size * sizeof(T),
                                   cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        throw std::runtime_error("CUDA copy failed: " +
                                 std::string(cudaGetErrorString(err)));
      }
    }
    return result;
  }

  /**
   * @brief Resize the buffer (reallocates memory)
   * @param new_size New size
   * @throws std::runtime_error if CUDA allocation fails
   */
  void resize(size_t new_size) {
    if (m_device_ptr != nullptr) {
      cudaFree(m_device_ptr);
      m_device_ptr = nullptr;
    }
    m_size = new_size;
    if (new_size > 0) {
      cudaError_t err = cudaMalloc(&m_device_ptr, new_size * sizeof(T));
      if (err != cudaSuccess) {
        throw std::runtime_error("CUDA allocation failed: " +
                                 std::string(cudaGetErrorString(err)));
      }
    }
  }
};
#endif // OpenPFC_ENABLE_CUDA

#if defined(OpenPFC_ENABLE_HIP)
/**
 * @brief HIP specialization of DataBuffer
 *
 * Uses HIP memory allocation (hipMalloc/hipFree).
 * Similar interface to CUDA version but uses HIP runtime.
 */
template <typename T> struct DataBuffer<backend::HipTag, T> {
private:
  T *m_device_ptr = nullptr;
  size_t m_size = 0;

public:
  /**
   * @brief Constructs a HIP buffer with the given size
   * @param size Number of elements
   * @throws std::runtime_error if HIP allocation fails
   */
  explicit DataBuffer(size_t size) : m_size(size) {
    if (size > 0) {
      hipError_t err = hipMalloc(&m_device_ptr, size * sizeof(T));
      if (err != hipSuccess) {
        throw std::runtime_error("HIP allocation failed: " +
                                 std::string(hipGetErrorString(err)));
      }
    }
  }

  /**
   * @brief Default constructor (empty buffer)
   */
  DataBuffer() = default;

  /**
   * @brief Destructor (frees HIP memory)
   */
  ~DataBuffer() {
    if (m_device_ptr != nullptr) {
      hipFree(m_device_ptr);
    }
  }

  /**
   * @brief Copy constructor is deleted (would require deep copy)
   */
  DataBuffer(const DataBuffer &) = delete;

  /**
   * @brief Copy assignment is deleted (would require deep copy)
   */
  DataBuffer &operator=(const DataBuffer &) = delete;

  /**
   * @brief Move constructor
   */
  DataBuffer(DataBuffer &&other) noexcept
      : m_device_ptr(other.m_device_ptr), m_size(other.m_size) {
    other.m_device_ptr = nullptr;
    other.m_size = 0;
  }

  /**
   * @brief Move assignment
   */
  DataBuffer &operator=(DataBuffer &&other) noexcept {
    if (this != &other) {
      if (m_device_ptr != nullptr) {
        hipFree(m_device_ptr);
      }
      m_device_ptr = other.m_device_ptr;
      m_size = other.m_size;
      other.m_device_ptr = nullptr;
      other.m_size = 0;
    }
    return *this;
  }

  /**
   * @brief Returns pointer to underlying data (device memory)
   * @return Pointer to data (device memory)
   */
  T *data() { return m_device_ptr; }

  /**
   * @brief Returns const pointer to underlying data (device memory)
   * @return Const pointer to data (device memory)
   */
  const T *data() const { return m_device_ptr; }

  /**
   * @brief Returns the number of elements
   * @return Size of buffer
   */
  size_t size() const { return m_size; }

  /**
   * @brief Returns true if buffer is empty
   * @return True if size() == 0
   */
  bool empty() const { return m_size == 0; }

  // Note: No operator[] - can't dereference device pointer on host!

  /**
   * @brief Copy data from host vector to device
   * @param src Source vector (must have same size)
   * @throws std::runtime_error if sizes don't match or copy fails
   */
  void copy_from_host(const std::vector<T> &src) {
    if (src.size() != m_size) {
      throw std::runtime_error("Size mismatch in copy_from_host: expected " +
                               std::to_string(m_size) + ", got " +
                               std::to_string(src.size()));
    }
    if (m_size > 0) {
      hipError_t err = hipMemcpy(m_device_ptr, src.data(), m_size * sizeof(T),
                                 hipMemcpyHostToDevice);
      if (err != hipSuccess) {
        throw std::runtime_error("HIP copy failed: " +
                                 std::string(hipGetErrorString(err)));
      }
    }
  }

  /**
   * @brief Copy data from device to host vector
   * @return Vector containing buffer data
   * @throws std::runtime_error if copy fails
   */
  std::vector<T> to_host() const {
    std::vector<T> result(m_size);
    if (m_size > 0) {
      hipError_t err = hipMemcpy(result.data(), m_device_ptr, m_size * sizeof(T),
                                 hipMemcpyDeviceToHost);
      if (err != hipSuccess) {
        throw std::runtime_error("HIP copy failed: " +
                                 std::string(hipGetErrorString(err)));
      }
    }
    return result;
  }

  /**
   * @brief Resize the buffer (reallocates memory)
   * @param new_size New size
   * @throws std::runtime_error if HIP allocation fails
   */
  void resize(size_t new_size) {
    if (m_device_ptr != nullptr) {
      hipFree(m_device_ptr);
      m_device_ptr = nullptr;
    }
    m_size = new_size;
    if (new_size > 0) {
      hipError_t err = hipMalloc(&m_device_ptr, new_size * sizeof(T));
      if (err != hipSuccess) {
        throw std::runtime_error("HIP allocation failed: " +
                                 std::string(hipGetErrorString(err)));
      }
    }
  }
};
#endif // OpenPFC_ENABLE_HIP

} // namespace core
} // namespace pfc
