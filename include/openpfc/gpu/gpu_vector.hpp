// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file gpu_vector.hpp
 * @brief Simple GPU memory container with RAII management
 *
 * @details
 * GPUVector provides a simple, RAII-style container for GPU memory that
 * automatically manages CUDA memory allocation and deallocation. It's designed
 * to be similar to std::vector in interface, but memory lives on the GPU.
 *
 * Key features:
 * - Automatic memory management (RAII)
 * - Fixed size (no resizing)
 * - CPU-GPU transfer methods
 * - Move semantics supported
 * - Copy semantics disabled (would require deep copy)
 *
 * This header must compile on systems without CUDA (e.g., AMD systems).
 * CUDA-specific code is guarded with #ifdef OPENPFC_ENABLE_CUDA.
 *
 * @code
 * #ifdef OPENPFC_ENABLE_CUDA
 *     pfc::gpu::GPUVector<double> vec(100);
 *     std::vector<double> host_data(100, 1.0);
 *     vec.copy_from_host(host_data);
 *     // ... use GPU memory ...
 *     std::vector<double> result = vec.to_host();
 * #endif
 * @endcode
 *
 * @see gpu/kernels_simple.hpp for GPU kernel operations
 * @see tungsten-gpu-implementation-plan.md for usage examples
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef PFC_GPU_VECTOR_HPP
#define PFC_GPU_VECTOR_HPP

#include <cstddef>
#include <stdexcept>
#include <vector>

// Only include CUDA headers if CUDA is enabled
#if defined(OPENPFC_ENABLE_CUDA)
#include <cuda_runtime.h>
#define PFC_GPU_CUDA_AVAILABLE 1
#else
#define PFC_GPU_CUDA_AVAILABLE 0
#endif

namespace pfc {
namespace gpu {

/**
 * @brief Simple GPU memory container (RAII)
 *
 * Similar to std::vector but memory lives on GPU.
 * Fixed size (no resizing).
 *
 * @tparam T Element type (must be trivially copyable)
 *
 * @note This class only works when CUDA is enabled at compile time.
 *       On systems without CUDA, the class exists but operations will fail
 *       at runtime (or compile-time if properly guarded).
 *
 * @warning Do not use this class directly on systems without CUDA.
 *          Always check OPENPFC_ENABLE_CUDA before using.
 */
template <typename T> class GPUVector {
private:
#if PFC_GPU_CUDA_AVAILABLE
  T *m_device_ptr = nullptr;
#else
  void *m_device_ptr = nullptr; // Placeholder for non-CUDA builds
#endif
  size_t m_size = 0;

public:
  /**
   * @brief Construct a GPUVector with the specified size
   *
   * Allocates GPU memory for `size` elements of type T.
   *
   * @param size Number of elements to allocate
   * @throws std::runtime_error if GPU memory allocation fails
   *
   * @note On systems without CUDA, this will throw at runtime.
   *       Always guard usage with #ifdef OPENPFC_ENABLE_CUDA.
   */
  explicit GPUVector(size_t size) : m_size(size) {
#if PFC_GPU_CUDA_AVAILABLE
    if (size > 0) {
      cudaError_t err = cudaMalloc(&m_device_ptr, size * sizeof(T));
      if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory: " +
                                 std::string(cudaGetErrorString(err)));
      }
    }
#else
    // On non-CUDA systems, this should not be called
    // But we provide a stub to allow compilation
    if (size > 0) {
      throw std::runtime_error("GPUVector: CUDA not enabled at compile time");
    }
#endif
  }

  /**
   * @brief Destructor - automatically frees GPU memory
   */
  ~GPUVector() {
#if PFC_GPU_CUDA_AVAILABLE
    if (m_device_ptr) {
      cudaFree(m_device_ptr);
    }
#endif
  }

  // No copy (would need deep copy)
  GPUVector(const GPUVector &) = delete;
  GPUVector &operator=(const GPUVector &) = delete;

  /**
   * @brief Move constructor
   */
  GPUVector(GPUVector &&other) noexcept
      : m_device_ptr(other.m_device_ptr), m_size(other.m_size) {
    other.m_device_ptr = nullptr;
    other.m_size = 0;
  }

  /**
   * @brief Move assignment operator
   */
  GPUVector &operator=(GPUVector &&other) noexcept {
    if (this != &other) {
#if PFC_GPU_CUDA_AVAILABLE
      if (m_device_ptr) cudaFree(m_device_ptr);
#endif
      m_device_ptr = other.m_device_ptr;
      m_size = other.m_size;
      other.m_device_ptr = nullptr;
      other.m_size = 0;
    }
    return *this;
  }

  /**
   * @brief Get pointer to GPU memory
   *
   * @return Pointer to device memory (nullptr if size is 0)
   *
   * @warning This pointer is only valid on the GPU. Do not dereference
   *          on the CPU. Use copy_from_host/copy_to_host for CPU access.
   */
  T *data() {
#if PFC_GPU_CUDA_AVAILABLE
    return m_device_ptr;
#else
    return nullptr;
#endif
  }

  /**
   * @brief Get const pointer to GPU memory
   */
  const T *data() const {
#if PFC_GPU_CUDA_AVAILABLE
    return m_device_ptr;
#else
    return nullptr;
#endif
  }

  /**
   * @brief Get the number of elements
   */
  size_t size() const { return m_size; }

  /**
   * @brief Check if the vector is empty
   */
  bool empty() const { return m_size == 0; }

  /**
   * @brief Copy data from host (CPU) to device (GPU)
   *
   * @param host_data Source data on CPU
   * @throws std::runtime_error if sizes don't match or copy fails
   */
  void copy_from_host(const std::vector<T> &host_data) {
    if (host_data.size() != m_size) {
      throw std::runtime_error("Size mismatch in copy_from_host: expected " +
                               std::to_string(m_size) + ", got " +
                               std::to_string(host_data.size()));
    }
#if PFC_GPU_CUDA_AVAILABLE
    if (m_size > 0) {
      cudaError_t err = cudaMemcpy(m_device_ptr, host_data.data(),
                                   m_size * sizeof(T), cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy from host to device: " +
                                 std::string(cudaGetErrorString(err)));
      }
    }
#else
    throw std::runtime_error("copy_from_host: CUDA not enabled at compile time");
#endif
  }

  /**
   * @brief Copy data from device (GPU) to host (CPU)
   *
   * @param host_data Destination vector (will be resized if needed)
   */
  void copy_to_host(std::vector<T> &host_data) const {
    if (host_data.size() != m_size) {
      host_data.resize(m_size);
    }
#if PFC_GPU_CUDA_AVAILABLE
    if (m_size > 0) {
      cudaError_t err = cudaMemcpy(host_data.data(), m_device_ptr,
                                   m_size * sizeof(T), cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy from device to host: " +
                                 std::string(cudaGetErrorString(err)));
      }
    }
#else
    throw std::runtime_error("copy_to_host: CUDA not enabled at compile time");
#endif
  }

  /**
   * @brief Copy data from device to host and return as vector
   *
   * @return std::vector<T> containing the data from GPU
   */
  std::vector<T> to_host() const {
    std::vector<T> result(m_size);
    copy_to_host(result);
    return result;
  }
};

} // namespace gpu
} // namespace pfc

#undef PFC_GPU_CUDA_AVAILABLE

#endif
