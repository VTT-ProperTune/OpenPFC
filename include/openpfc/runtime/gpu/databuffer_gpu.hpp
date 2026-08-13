// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file databuffer_gpu.hpp
 * @brief Single-source GPU `DataBuffer` for CUDA and HIP (M3).
 *
 * One implementation stamps `DataBuffer<CudaTag, T>` and/or
 * `DataBuffer<HipTag, T>` depending on the enabled backends. Vendor headers
 * `runtime/cuda/databuffer_cuda.hpp` and `runtime/hip/databuffer_hip.hpp` are
 * thin includes of this file so existing call sites keep compiling.
 *
 * Per-tag allocators call the native runtime (not `gpu_api.hpp`) so a
 * CUDA+HIP co-enabled translation unit can own both specializations.
 *
 * @see kernel/execution/databuffer.hpp
 * @see runtime/gpu/gpu_api.hpp
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/kernel/execution/databuffer.hpp>
#include <openpfc/runtime/gpu/backend_tags_gpu.hpp>

#if defined(OpenPFC_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif
#if defined(OpenPFC_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace pfc::core::detail {

#if defined(OpenPFC_ENABLE_CUDA)
struct CudaAlloc {
  using error_t = cudaError_t;
  static constexpr error_t success = cudaSuccess;
  static constexpr const char *kind = "CUDA";
  static error_t malloc(void **ptr, std::size_t bytes) {
    return cudaMalloc(ptr, bytes);
  }
  static error_t free(void *ptr) { return cudaFree(ptr); }
  static error_t memcpy_h2d(void *dst, const void *src, std::size_t bytes) {
    return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
  }
  static error_t memcpy_d2h(void *dst, const void *src, std::size_t bytes) {
    return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
  }
  static error_t get_last_error() { return cudaGetLastError(); }
  static const char *error_string(error_t err) { return cudaGetErrorString(err); }
};
#endif

#if defined(OpenPFC_ENABLE_HIP)
struct HipAlloc {
  using error_t = hipError_t;
  static constexpr error_t success = hipSuccess;
  static constexpr const char *kind = "HIP";
  static error_t malloc(void **ptr, std::size_t bytes) {
    return hipMalloc(ptr, bytes);
  }
  static error_t free(void *ptr) { return hipFree(ptr); }
  static error_t memcpy_h2d(void *dst, const void *src, std::size_t bytes) {
    return hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
  }
  static error_t memcpy_d2h(void *dst, const void *src, std::size_t bytes) {
    return hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
  }
  static error_t get_last_error() { return hipGetLastError(); }
  static const char *error_string(error_t err) { return hipGetErrorString(err); }
};
#endif

template <typename T, typename Alloc> struct GpuDataBuffer {
private:
  T *m_device_ptr = nullptr;
  std::size_t m_size = 0;

  static void throw_on_error(typename Alloc::error_t err, const char *op) {
    [[maybe_unused]] const auto cleared = Alloc::get_last_error();
    throw std::runtime_error(std::string(Alloc::kind) + " " + op + ": " +
                             Alloc::error_string(err));
  }

public:
  explicit GpuDataBuffer(std::size_t size) : m_size(size) {
    if (size > 0) {
      auto err = Alloc::malloc(reinterpret_cast<void **>(&m_device_ptr),
                               size * sizeof(T));
      if (err != Alloc::success) {
        throw_on_error(err, "allocation failed");
      }
    }
  }

  GpuDataBuffer() = default;

  ~GpuDataBuffer() {
    if (m_device_ptr != nullptr) {
      [[maybe_unused]] const auto freed = Alloc::free(m_device_ptr);
    }
  }

  GpuDataBuffer(const GpuDataBuffer &) = delete;
  GpuDataBuffer &operator=(const GpuDataBuffer &) = delete;

  GpuDataBuffer(GpuDataBuffer &&other) noexcept
      : m_device_ptr(other.m_device_ptr), m_size(other.m_size) {
    other.m_device_ptr = nullptr;
    other.m_size = 0;
  }

  GpuDataBuffer &operator=(GpuDataBuffer &&other) noexcept {
    if (this != &other) {
      if (m_device_ptr != nullptr) {
        [[maybe_unused]] const auto freed = Alloc::free(m_device_ptr);
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
  std::size_t size() const { return m_size; }
  bool empty() const { return m_size == 0; }

  void copy_from_host(const std::vector<T> &src) {
    copy_from_host(src.data(), src.size());
  }

  void copy_from_host(const T *ptr, std::size_t n) {
    if (n != m_size) {
      throw std::runtime_error("Size mismatch in copy_from_host: expected " +
                               std::to_string(m_size) + ", got " +
                               std::to_string(n));
    }
    if (m_size > 0) {
      auto err = Alloc::memcpy_h2d(m_device_ptr, ptr, m_size * sizeof(T));
      if (err != Alloc::success) {
        throw_on_error(err, "copy failed");
      }
    }
  }

  void copy_from_host(std::span<const T> src) {
    copy_from_host(src.data(), src.size());
  }

  void copy_to_host(T *ptr, std::size_t n) const {
    if (n != m_size) {
      throw std::runtime_error("Size mismatch in copy_to_host: expected " +
                               std::to_string(m_size) + ", got " +
                               std::to_string(n));
    }
    if (m_size > 0) {
      auto err = Alloc::memcpy_d2h(ptr, m_device_ptr, m_size * sizeof(T));
      if (err != Alloc::success) {
        throw_on_error(err, "copy failed");
      }
    }
  }

  std::vector<T> to_host() const {
    std::vector<T> result(m_size);
    if (m_size > 0) {
      auto err =
          Alloc::memcpy_d2h(result.data(), m_device_ptr, m_size * sizeof(T));
      if (err != Alloc::success) {
        throw_on_error(err, "copy failed");
      }
    }
    return result;
  }

  void resize(std::size_t new_size) {
    if (new_size == 0) {
      if (m_device_ptr != nullptr) {
        [[maybe_unused]] const auto freed = Alloc::free(m_device_ptr);
        m_device_ptr = nullptr;
      }
      m_size = 0;
      return;
    }

    T *new_ptr = nullptr;
    auto err = Alloc::malloc(reinterpret_cast<void **>(&new_ptr),
                             new_size * sizeof(T));
    if (err != Alloc::success) {
      throw_on_error(err, "allocation failed");
    }

    if (m_device_ptr != nullptr) {
      [[maybe_unused]] const auto freed = Alloc::free(m_device_ptr);
    }
    m_device_ptr = new_ptr;
    m_size = new_size;
  }
};

} // namespace pfc::core::detail

namespace pfc::core {

#if defined(OpenPFC_ENABLE_CUDA)
template <typename T>
struct DataBuffer<backend::CudaTag, T>
    : detail::GpuDataBuffer<T, detail::CudaAlloc> {
  using detail::GpuDataBuffer<T, detail::CudaAlloc>::GpuDataBuffer;
};
#endif

#if defined(OpenPFC_ENABLE_HIP)
template <typename T>
struct DataBuffer<backend::HipTag, T>
    : detail::GpuDataBuffer<T, detail::HipAlloc> {
  using detail::GpuDataBuffer<T, detail::HipAlloc>::GpuDataBuffer;
};
#endif

} // namespace pfc::core

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
