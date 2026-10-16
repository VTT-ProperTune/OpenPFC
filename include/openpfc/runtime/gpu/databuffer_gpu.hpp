// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file databuffer_gpu.hpp
 * @brief Single-source GPU `DataBuffer` for CUDA and HIP (M3).
 *
 * One implementation stamps `DataBuffer<CUDATag, T>` and/or
 * `DataBuffer<HIPTag, T>` depending on the enabled backends. Vendor headers
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
#include <limits>
#include <openpfc/kernel/grain/diagnostics.hpp>
#include <optional>
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
struct CUDAAlloc {
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
  static error_t memset_zero(void *dst, std::size_t bytes) {
    return cudaMemset(dst, 0, bytes);
  }
  static error_t get_last_error() { return cudaGetLastError(); }
  static const char *error_string(error_t err) { return cudaGetErrorString(err); }
};
#endif

#if defined(OpenPFC_ENABLE_HIP)
struct HIPAlloc {
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
  static error_t memset_zero(void *dst, std::size_t bytes) {
    return hipMemset(dst, 0, bytes);
  }
  static error_t get_last_error() { return hipGetLastError(); }
  static const char *error_string(error_t err) { return hipGetErrorString(err); }
};
#endif

template <typename T, typename Alloc> struct GPUDataBuffer {
private:
  T *m_device_ptr = nullptr;
  std::size_t m_size = 0;
  std::optional<pfc::grain::diagnostics::AllocationToken> m_allocation;

  static void throw_on_error(typename Alloc::error_t err, const char *op) {
    [[maybe_unused]] const auto cleared = Alloc::get_last_error();
    throw std::runtime_error(std::string(Alloc::kind) + " " + op + ": " +
                             Alloc::error_string(err));
  }

public:
  explicit GPUDataBuffer(std::size_t size) : m_size(size) {
    if (size > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::overflow_error("allocation payload size overflow");
    if (size > 0) {
      auto err =
          Alloc::malloc(reinterpret_cast<void **>(&m_device_ptr), size * sizeof(T));
      if (err != Alloc::success) {
        throw_on_error(err, "allocation failed");
      }
      m_allocation.emplace(pfc::grain::diagnostics::successful_allocation(
          pfc::grain::diagnostics::Space::Device, size * sizeof(T)));
    }
  }

  GPUDataBuffer() = default;

  ~GPUDataBuffer() {
    if (m_device_ptr != nullptr) {
      const auto freed = Alloc::free(m_device_ptr);
      if (m_allocation) {
        if (freed == Alloc::success)
          m_allocation->release();
        else
          m_allocation->failed_release();
      }
    }
  }

  GPUDataBuffer(const GPUDataBuffer &) = delete;
  GPUDataBuffer &operator=(const GPUDataBuffer &) = delete;

  GPUDataBuffer(GPUDataBuffer &&other) noexcept
      : m_device_ptr(other.m_device_ptr), m_size(other.m_size),
        m_allocation(std::move(other.m_allocation)) {
    other.m_device_ptr = nullptr;
    other.m_size = 0;
  }

  GPUDataBuffer &operator=(GPUDataBuffer &&other) noexcept {
    if (this != &other) {
      if (m_device_ptr != nullptr) {
        const auto freed = Alloc::free(m_device_ptr);
        if (m_allocation) {
          if (freed == Alloc::success)
            m_allocation->release();
          else
            m_allocation->failed_release();
        }
      }
      m_allocation.reset();
      if (other.m_allocation) m_allocation.emplace(std::move(*other.m_allocation));
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

  /// Set every element to all-zero bytes (0 for arithmetic and complex types).
  void zero() {
    if (m_size > 0) {
      auto err = Alloc::memset_zero(m_device_ptr, m_size * sizeof(T));
      if (err != Alloc::success) {
        throw_on_error(err, "memset failed");
      }
    }
  }

  void copy_from_host(
      const std::vector<T> &src,
      pfc::grain::diagnostics::Field field = pfc::grain::diagnostics::Field::Other) {
    copy_from_host(src.data(), src.size(), field);
  }

  void copy_from_host(
      const T *ptr, std::size_t n,
      pfc::grain::diagnostics::Field field = pfc::grain::diagnostics::Field::Other) {
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
      pfc::grain::diagnostics::copy(pfc::grain::diagnostics::Direction::HostToDevice,
                                    field, m_size * sizeof(T));
    }
  }

  void copy_from_host(
      std::span<const T> src,
      pfc::grain::diagnostics::Field field = pfc::grain::diagnostics::Field::Other) {
    copy_from_host(src.data(), src.size(), field);
  }

  void copy_to_host(T *ptr, std::size_t n,
                    pfc::grain::diagnostics::Field field =
                        pfc::grain::diagnostics::Field::Other) const {
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
      pfc::grain::diagnostics::copy(pfc::grain::diagnostics::Direction::DeviceToHost,
                                    field, m_size * sizeof(T));
    }
  }

  std::vector<T> to_host(pfc::grain::diagnostics::Field field =
                             pfc::grain::diagnostics::Field::Other) const {
    std::vector<T> result(m_size);
    if (m_size > 0) {
      auto err = Alloc::memcpy_d2h(result.data(), m_device_ptr, m_size * sizeof(T));
      if (err != Alloc::success) {
        throw_on_error(err, "copy failed");
      }
      pfc::grain::diagnostics::copy(pfc::grain::diagnostics::Direction::DeviceToHost,
                                    field, m_size * sizeof(T));
    }
    return result;
  }

  void resize(std::size_t new_size) {
    if (new_size > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::overflow_error("allocation payload size overflow");
    if (new_size == 0) {
      if (m_device_ptr != nullptr) {
        const auto freed = Alloc::free(m_device_ptr);
        if (m_allocation) {
          if (freed == Alloc::success)
            m_allocation->release();
          else
            m_allocation->failed_release();
        }
        m_device_ptr = nullptr;
      }
      m_size = 0;
      return;
    }

    T *new_ptr = nullptr;
    auto err =
        Alloc::malloc(reinterpret_cast<void **>(&new_ptr), new_size * sizeof(T));
    if (err != Alloc::success) {
      throw_on_error(err, "allocation failed");
    }

    auto allocation = pfc::grain::diagnostics::successful_allocation(
        pfc::grain::diagnostics::Space::Device, new_size * sizeof(T));
    if (m_device_ptr != nullptr) {
      const auto freed = Alloc::free(m_device_ptr);
      if (m_allocation) {
        if (freed == Alloc::success)
          m_allocation->release();
        else
          m_allocation->failed_release();
      }
    }
    m_allocation.reset();
    m_allocation.emplace(std::move(allocation));
    m_device_ptr = new_ptr;
    m_size = new_size;
  }
};

} // namespace pfc::core::detail

namespace pfc::core {

#if defined(OpenPFC_ENABLE_CUDA)
template <typename T>
struct DataBuffer<backend::CUDATag, T>
    : detail::GPUDataBuffer<T, detail::CUDAAlloc> {
  using detail::GPUDataBuffer<T, detail::CUDAAlloc>::GPUDataBuffer;
};
#endif

#if defined(OpenPFC_ENABLE_HIP)
template <typename T>
struct DataBuffer<backend::HIPTag, T> : detail::GPUDataBuffer<T, detail::HIPAlloc> {
  using detail::GPUDataBuffer<T, detail::HIPAlloc>::GPUDataBuffer;
};
#endif

} // namespace pfc::core

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
