// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file databuffer_cuda.hpp
 * @brief CUDA specialization of DataBuffer (runtime/cuda only)
 *
 * Include this header when using DataBuffer<CudaTag, T>. Kernel defines only
 * CpuTag; CUDA and HIP specializations live in runtime.
 *
 * @see kernel/execution/databuffer.hpp for CpuTag and interface
 * @see runtime/hip/databuffer_hip.hpp for HipTag
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA)

#include <cstddef>
#include <cuda_runtime.h>
#include <openpfc/kernel/execution/databuffer.hpp>
#include <openpfc/runtime/cuda/backend_tags_cuda.hpp>
#include <span>
#include <stdexcept>
#include <vector>

namespace pfc {
namespace core {

/**
 * @brief CUDA specialization of DataBuffer
 *
 * Uses CUDA memory allocation (cudaMalloc/cudaFree). Does not provide
 * operator[] (can't dereference device pointer on host).
 */
template <typename T> struct DataBuffer<backend::CudaTag, T> {
private:
  T *m_device_ptr = nullptr;
  size_t m_size = 0;

public:
  explicit DataBuffer(size_t size) : m_size(size) {
    if (size > 0) {
      cudaError_t err = cudaMalloc(&m_device_ptr, size * sizeof(T));
      if (err != cudaSuccess) {
        throw std::runtime_error("CUDA allocation failed: " +
                                 std::string(cudaGetErrorString(err)));
      }
    }
  }

  DataBuffer() = default;

  ~DataBuffer() {
    if (m_device_ptr != nullptr) {
      cudaFree(m_device_ptr);
    }
  }

  DataBuffer(const DataBuffer &) = delete;
  DataBuffer &operator=(const DataBuffer &) = delete;

  DataBuffer(DataBuffer &&other) noexcept
      : m_device_ptr(other.m_device_ptr), m_size(other.m_size) {
    other.m_device_ptr = nullptr;
    other.m_size = 0;
  }

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

  T *data() { return m_device_ptr; }
  const T *data() const { return m_device_ptr; }
  size_t size() const { return m_size; }
  bool empty() const { return m_size == 0; }

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

  void copy_from_host(const T *ptr, size_t n) {
    if (n != m_size) {
      throw std::runtime_error("Size mismatch in copy_from_host: expected " +
                               std::to_string(m_size) + ", got " +
                               std::to_string(n));
    }
    if (m_size > 0) {
      cudaError_t err =
          cudaMemcpy(m_device_ptr, ptr, m_size * sizeof(T), cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        throw std::runtime_error("CUDA copy failed: " +
                                 std::string(cudaGetErrorString(err)));
      }
    }
  }

  void copy_from_host(std::span<const T> src) {
    copy_from_host(src.data(), src.size());
  }

  void copy_to_host(T *ptr, size_t n) const {
    if (n != m_size) {
      throw std::runtime_error("Size mismatch in copy_to_host: expected " +
                               std::to_string(m_size) + ", got " +
                               std::to_string(n));
    }
    if (m_size > 0) {
      cudaError_t err =
          cudaMemcpy(ptr, m_device_ptr, m_size * sizeof(T), cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        throw std::runtime_error("CUDA copy failed: " +
                                 std::string(cudaGetErrorString(err)));
      }
    }
  }

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

} // namespace core
} // namespace pfc

#endif // OpenPFC_ENABLE_CUDA
