// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file databuffer_gpu.hpp
 * @brief Single-source GPU DataBuffer specialization (M3)
 *
 * Unified implementation for CUDA and HIP backends using the GPU vendor shim.
 * Replaces runtime/cuda/databuffer_cuda.hpp and runtime/hip/databuffer_hip.hpp.
 *
 * @see kernel/execution/databuffer.hpp for CpuTag and interface
 * @see runtime/gpu/gpu_api.hpp for GPU API vendor shim
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <cstddef>
#include <openpfc/kernel/execution/databuffer.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <span>
#include <stdexcept>
#include <vector>

#if defined(OpenPFC_ENABLE_CUDA)
#include <openpfc/runtime/cuda/backend_tags_cuda.hpp>
#elif defined(OpenPFC_ENABLE_HIP)
#include <openpfc/runtime/hip/backend_tags_hip.hpp>
#endif

namespace pfc {
namespace core {

#if defined(OpenPFC_ENABLE_CUDA)

/**
 * @brief CUDA specialization of DataBuffer
 *
 * Uses CUDA memory allocation via unified GPU API. Does not provide
 * operator[] (can't dereference device pointer on host).
 */
template <typename T> struct DataBuffer<backend::CudaTag, T> {
private:
  T *m_device_ptr = nullptr;
  size_t m_size = 0;

public:
  explicit DataBuffer(size_t size) : m_size(size) {
    if (size > 0) {
      gpuError_t err =
          gpuMalloc(reinterpret_cast<void **>(&m_device_ptr), size * sizeof(T));
      if (err != gpuSuccess) {
        // Consume the sticky driver-level error now, on our own terms —
        // otherwise it survives as the "last error" and gets misattributed
        // to the next unrelated call anywhere else in the process.
        gpuGetLastError();
        throw std::runtime_error("GPU allocation failed: " +
                                 std::string(gpuGetErrorString(err)));
      }
    }
  }

  DataBuffer() = default;

  ~DataBuffer() {
    if (m_device_ptr != nullptr) {
      [[maybe_unused]] const gpuError_t freed = gpuFree(m_device_ptr);
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
        [[maybe_unused]] const gpuError_t freed = gpuFree(m_device_ptr);
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
      gpuError_t err = gpuMemcpy(m_device_ptr, src.data(), m_size * sizeof(T),
                                 gpuMemcpyHostToDevice);
      if (err != gpuSuccess) {
        gpuGetLastError(); // see DataBuffer(size_t) ctor: clear sticky error
        throw std::runtime_error("GPU copy failed: " +
                                 std::string(gpuGetErrorString(err)));
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
      gpuError_t err =
          gpuMemcpy(m_device_ptr, ptr, m_size * sizeof(T), gpuMemcpyHostToDevice);
      if (err != gpuSuccess) {
        gpuGetLastError(); // see DataBuffer(size_t) ctor: clear sticky error
        throw std::runtime_error("GPU copy failed: " +
                                 std::string(gpuGetErrorString(err)));
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
      gpuError_t err =
          gpuMemcpy(ptr, m_device_ptr, m_size * sizeof(T), gpuMemcpyDeviceToHost);
      if (err != gpuSuccess) {
        gpuGetLastError(); // see DataBuffer(size_t) ctor: clear sticky error
        throw std::runtime_error("GPU copy failed: " +
                                 std::string(gpuGetErrorString(err)));
      }
    }
  }

  std::vector<T> to_host() const {
    std::vector<T> result(m_size);
    if (m_size > 0) {
      gpuError_t err = gpuMemcpy(result.data(), m_device_ptr, m_size * sizeof(T),
                                 gpuMemcpyDeviceToHost);
      if (err != gpuSuccess) {
        gpuGetLastError(); // see DataBuffer(size_t) ctor: clear sticky error
        throw std::runtime_error("GPU copy failed: " +
                                 std::string(gpuGetErrorString(err)));
      }
    }
    return result;
  }

  /**
   * @brief Resize the device buffer.
   *
   * Allocates the new buffer before freeing the old one. On allocation
   * failure, size() and data() remain unchanged. After a successful alloc,
   * the new buffer is published even if freeing the previous pointer fails
   * (best-effort free; peak device memory briefly doubles during grow).
   */
  void resize(size_t new_size) {
    if (new_size == 0) {
      if (m_device_ptr != nullptr) {
        [[maybe_unused]] const gpuError_t freed = gpuFree(m_device_ptr);
        m_device_ptr = nullptr;
      }
      m_size = 0;
      return;
    }

    void *new_ptr = nullptr;
    gpuError_t err = gpuMalloc(&new_ptr, new_size * sizeof(T));
    if (err != gpuSuccess) {
      gpuGetLastError(); // see DataBuffer(size_t) ctor: clear sticky error
      throw std::runtime_error("GPU allocation failed: " +
                               std::string(gpuGetErrorString(err)));
    }

    if (m_device_ptr != nullptr) {
      [[maybe_unused]] const gpuError_t freed = gpuFree(m_device_ptr);
    }
    m_device_ptr = static_cast<T *>(new_ptr);
    m_size = new_size;
  }
};

#elif defined(OpenPFC_ENABLE_HIP)

/**
 * @brief HIP specialization of DataBuffer
 *
 * Uses HIP memory allocation via unified GPU API. Same interface as CUDA version.
 */
template <typename T> struct DataBuffer<backend::HipTag, T> {
private:
  T *m_device_ptr = nullptr;
  size_t m_size = 0;

public:
  explicit DataBuffer(size_t size) : m_size(size) {
    if (size > 0) {
      gpuError_t err =
          gpuMalloc(reinterpret_cast<void **>(&m_device_ptr), size * sizeof(T));
      if (err != gpuSuccess) {
        // Consume the sticky driver-level error now, on our own terms —
        // otherwise it survives as the "last error" and gets misattributed
        // to the next unrelated call anywhere else in the process.
        gpuGetLastError();
        throw std::runtime_error("GPU allocation failed: " +
                                 std::string(gpuGetErrorString(err)));
      }
    }
  }

  DataBuffer() = default;

  ~DataBuffer() {
    if (m_device_ptr != nullptr) {
      [[maybe_unused]] const gpuError_t freed = gpuFree(m_device_ptr);
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
        [[maybe_unused]] const gpuError_t freed = gpuFree(m_device_ptr);
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
      gpuError_t err = gpuMemcpy(m_device_ptr, src.data(), m_size * sizeof(T),
                                 gpuMemcpyHostToDevice);
      if (err != gpuSuccess) {
        gpuGetLastError(); // see DataBuffer(size_t) ctor: clear sticky error
        throw std::runtime_error("GPU copy failed: " +
                                 std::string(gpuGetErrorString(err)));
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
      gpuError_t err =
          gpuMemcpy(m_device_ptr, ptr, m_size * sizeof(T), gpuMemcpyHostToDevice);
      if (err != gpuSuccess) {
        gpuGetLastError(); // see DataBuffer(size_t) ctor: clear sticky error
        throw std::runtime_error("GPU copy failed: " +
                                 std::string(gpuGetErrorString(err)));
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
      gpuError_t err =
          gpuMemcpy(ptr, m_device_ptr, m_size * sizeof(T), gpuMemcpyDeviceToHost);
      if (err != gpuSuccess) {
        gpuGetLastError(); // see DataBuffer(size_t) ctor: clear sticky error
        throw std::runtime_error("GPU copy failed: " +
                                 std::string(gpuGetErrorString(err)));
      }
    }
  }

  std::vector<T> to_host() const {
    std::vector<T> result(m_size);
    if (m_size > 0) {
      gpuError_t err = gpuMemcpy(result.data(), m_device_ptr, m_size * sizeof(T),
                                 gpuMemcpyDeviceToHost);
      if (err != gpuSuccess) {
        gpuGetLastError(); // see DataBuffer(size_t) ctor: clear sticky error
        throw std::runtime_error("GPU copy failed: " +
                                 std::string(gpuGetErrorString(err)));
      }
    }
    return result;
  }

  /**
   * @brief Resize the device buffer.
   *
   * Allocates the new buffer before freeing the old one. On allocation
   * failure, size() and data() remain unchanged. After a successful alloc,
   * the new buffer is published even if freeing the previous pointer fails
   * (best-effort free; peak device memory briefly doubles during grow).
   */
  void resize(size_t new_size) {
    if (new_size == 0) {
      if (m_device_ptr != nullptr) {
        [[maybe_unused]] const gpuError_t freed = gpuFree(m_device_ptr);
        m_device_ptr = nullptr;
      }
      m_size = 0;
      return;
    }

    void *new_ptr = nullptr;
    gpuError_t err = gpuMalloc(&new_ptr, new_size * sizeof(T));
    if (err != gpuSuccess) {
      gpuGetLastError(); // see DataBuffer(size_t) ctor: clear sticky error
      throw std::runtime_error("GPU allocation failed: " +
                               std::string(gpuGetErrorString(err)));
    }

    if (m_device_ptr != nullptr) {
      [[maybe_unused]] const gpuError_t freed = gpuFree(m_device_ptr);
    }
    m_device_ptr = static_cast<T *>(new_ptr);
    m_size = new_size;
  }
};

#endif

} // namespace core
} // namespace pfc

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP