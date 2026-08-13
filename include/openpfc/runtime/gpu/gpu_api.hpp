// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file gpu_api.hpp
 * @brief Vendor shim for single-source CUDA/HIP runtime calls (M3).
 *
 * Include this from `runtime/gpu/` sources instead of `<cuda_runtime.h>` or
 * `<hip/hip_runtime.h>`. Names match the CUDA vocabulary with a `gpu` prefix
 * (`gpuMalloc`, `gpuMemcpyAsync`, `gpuStream_t`, `gpuEvent_t`).
 *
 * Backend selection:
 *   - hipcc / `__HIPCC__` / `__HIP__` → HIP (checked first; hipcc may also
 *     define `__CUDACC__`)
 *   - nvcc / `__CUDACC__` / `OpenPFC_ENABLE_CUDA` → CUDA
 *   - `OpenPFC_ENABLE_HIP` on a host TU → HIP
 *
 * `OPENPFC_HD` lives in `kernel/data/host_device.hpp` (already has an
 * explicit `__HIPCC__` branch). This header does not redefine it.
 *
 * @see kernel/data/host_device.hpp
 */

#include <cstddef>
#include <stdexcept>
#include <string>

#if defined(__HIPCC__) || defined(__HIP__)
#define OPENPFC_GPU_API_HIP 1
#include <hip/hip_runtime.h>
#elif defined(__CUDACC__) || defined(OpenPFC_ENABLE_CUDA)
#define OPENPFC_GPU_API_CUDA 1
#include <cuda_runtime.h>
#elif defined(OpenPFC_ENABLE_HIP)
#define OPENPFC_GPU_API_HIP 1
#include <hip/hip_runtime.h>
#else
#error "gpu_api.hpp requires OpenPFC_ENABLE_CUDA or OpenPFC_ENABLE_HIP"
#endif

namespace pfc {

#if defined(OPENPFC_GPU_API_CUDA)

using gpuStream_t = cudaStream_t;
using gpuEvent_t = cudaEvent_t;
using gpuError_t = cudaError_t;
using gpuMemcpyKind = cudaMemcpyKind;

inline constexpr gpuError_t gpuSuccess = cudaSuccess;
inline constexpr gpuError_t gpuErrorMemoryAllocation = cudaErrorMemoryAllocation;
inline constexpr gpuMemcpyKind gpuMemcpyHostToDevice = cudaMemcpyHostToDevice;
inline constexpr gpuMemcpyKind gpuMemcpyDeviceToHost = cudaMemcpyDeviceToHost;
inline constexpr gpuMemcpyKind gpuMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;
inline constexpr gpuMemcpyKind gpuMemcpyDefault = cudaMemcpyDefault;

inline gpuError_t gpuMalloc(void **ptr, std::size_t size) {
  return cudaMalloc(ptr, size);
}
inline gpuError_t gpuFree(void *ptr) { return cudaFree(ptr); }
inline gpuError_t gpuMemcpyAsync(void *dst, const void *src, std::size_t size,
                                 gpuMemcpyKind kind, gpuStream_t stream) {
  return cudaMemcpyAsync(dst, src, size, kind, stream);
}
inline gpuError_t gpuMemcpy(void *dst, const void *src, std::size_t size,
                            gpuMemcpyKind kind) {
  return cudaMemcpy(dst, src, size, kind);
}
inline gpuError_t gpuStreamCreate(gpuStream_t *stream) {
  return cudaStreamCreate(stream);
}
inline gpuError_t gpuStreamDestroy(gpuStream_t stream) {
  return cudaStreamDestroy(stream);
}
inline gpuError_t gpuStreamSynchronize(gpuStream_t stream) {
  return cudaStreamSynchronize(stream);
}
inline gpuError_t gpuEventCreate(gpuEvent_t *event) {
  return cudaEventCreate(event);
}
inline gpuError_t gpuEventDestroy(gpuEvent_t event) {
  return cudaEventDestroy(event);
}
inline gpuError_t gpuEventRecord(gpuEvent_t event, gpuStream_t stream) {
  return cudaEventRecord(event, stream);
}
inline gpuError_t gpuEventSynchronize(gpuEvent_t event) {
  return cudaEventSynchronize(event);
}
inline gpuError_t gpuDeviceSynchronize() { return cudaDeviceSynchronize(); }
inline gpuError_t gpuGetLastError() { return cudaGetLastError(); }
inline const char *gpuGetErrorString(gpuError_t error) {
  return cudaGetErrorString(error);
}

#elif defined(OPENPFC_GPU_API_HIP)

using gpuStream_t = hipStream_t;
using gpuEvent_t = hipEvent_t;
using gpuError_t = hipError_t;
using gpuMemcpyKind = hipMemcpyKind;

inline constexpr gpuError_t gpuSuccess = hipSuccess;
inline constexpr gpuError_t gpuErrorMemoryAllocation = hipErrorMemoryAllocation;
inline constexpr gpuMemcpyKind gpuMemcpyHostToDevice = hipMemcpyHostToDevice;
inline constexpr gpuMemcpyKind gpuMemcpyDeviceToHost = hipMemcpyDeviceToHost;
inline constexpr gpuMemcpyKind gpuMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
inline constexpr gpuMemcpyKind gpuMemcpyDefault = hipMemcpyDefault;

inline gpuError_t gpuMalloc(void **ptr, std::size_t size) {
  return hipMalloc(ptr, size);
}
inline gpuError_t gpuFree(void *ptr) { return hipFree(ptr); }
inline gpuError_t gpuMemcpyAsync(void *dst, const void *src, std::size_t size,
                                 gpuMemcpyKind kind, gpuStream_t stream) {
  return hipMemcpyAsync(dst, src, size, kind, stream);
}
inline gpuError_t gpuMemcpy(void *dst, const void *src, std::size_t size,
                            gpuMemcpyKind kind) {
  return hipMemcpy(dst, src, size, kind);
}
inline gpuError_t gpuStreamCreate(gpuStream_t *stream) {
  return hipStreamCreate(stream);
}
inline gpuError_t gpuStreamDestroy(gpuStream_t stream) {
  return hipStreamDestroy(stream);
}
inline gpuError_t gpuStreamSynchronize(gpuStream_t stream) {
  return hipStreamSynchronize(stream);
}
inline gpuError_t gpuEventCreate(gpuEvent_t *event) { return hipEventCreate(event); }
inline gpuError_t gpuEventDestroy(gpuEvent_t event) { return hipEventDestroy(event); }
inline gpuError_t gpuEventRecord(gpuEvent_t event, gpuStream_t stream) {
  return hipEventRecord(event, stream);
}
inline gpuError_t gpuEventSynchronize(gpuEvent_t event) {
  return hipEventSynchronize(event);
}
inline gpuError_t gpuDeviceSynchronize() { return hipDeviceSynchronize(); }
inline gpuError_t gpuGetLastError() { return hipGetLastError(); }
inline const char *gpuGetErrorString(gpuError_t error) {
  return hipGetErrorString(error);
}

#endif

} // namespace pfc

/// Throw `std::runtime_error` naming the GPU status if `call` is not success.
#define GPU_CHECK(call)                                                             \
  do {                                                                              \
    ::pfc::gpuError_t gpu_check_status_ = (call);                                   \
    if (gpu_check_status_ != ::pfc::gpuSuccess) {                                   \
      throw std::runtime_error(std::string("GPU error: ") +                         \
                               ::pfc::gpuGetErrorString(gpu_check_status_));        \
    }                                                                               \
  } while (0)

/// Launch `kernel` with `<<<grid, block, 0, stream>>>(...)`.
/// `args` is the parenthesized argument list, e.g. `(a, b, c)`.
#define GPU_LAUNCH_KERNEL(kernel, grid, block, args, stream)                        \
  do {                                                                              \
    kernel<<<(grid), (block), 0, (stream)>>> args;                                  \
    GPU_CHECK(::pfc::gpuGetLastError());                                            \
  } while (0)
