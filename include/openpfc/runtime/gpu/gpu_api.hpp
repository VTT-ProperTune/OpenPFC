// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file gpu_api.hpp
 * @brief Vendor-shim GPU API for single-source CUDA/HIP code (M3)
 *
 * This header provides unified GPU API functions that map to either CUDA or HIP
 * based on the build configuration (OpenPFC_ENABLE_CUDA or OpenPFC_ENABLE_HIP).
 * It enables single-source GPU code that compiles for both backends.
 *
 * Usage: Include this header instead of cuda_runtime.h or hip_runtime.h in
 * runtime/gpu/ single-source files. Use gpu* prefixes for all GPU API calls.
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA)

// Include CUDA headers when CUDA is enabled
#include <cuda_runtime.h>

#define GPU_CHECK(call)                                                             \
  do {                                                                              \
    cudaError_t error = call;                                                       \
    if (error != cudaSuccess) {                                                     \
      throw std::runtime_error(std::string("GPU error: ") +                         \
                               cudaGetErrorString(error));                          \
    }                                                                               \
  } while (0)

// Type mappings
using gpuStream_t = cudaStream_t;
using gpuEvent_t = cudaEvent_t;
using gpuError_t = cudaError_t;
using cudaMemcpyKind = cudaMemcpyKind;

// Function mappings
static constexpr auto gpuSuccess = cudaSuccess;
static constexpr auto gpuErrorMemoryAllocation = cudaErrorMemoryAllocation;
static constexpr auto gpuMemcpyHostToDevice = cudaMemcpyHostToDevice;
static constexpr auto gpuMemcpyDeviceToHost = cudaMemcpyDeviceToDevice;
static constexpr auto gpuMemcpyDeviceToDevice = cudaMemcpyDeviceToHost;
static constexpr auto gpuMemcpyDefault = cudaMemcpyDefault;

// Function wrappers
inline gpuError_t gpuMalloc(void **ptr, size_t size) {
  return cudaMalloc(ptr, size);
}

inline gpuError_t gpuFree(void *ptr) { return cudaFree(ptr); }

inline gpuError_t gpuMemcpyAsync(void *dst, const void *src, size_t size,
                                 cudaMemcpyKind kind, gpuStream_t stream) {
  return cudaMemcpyAsync(dst, src, size, kind, stream);
}

inline gpuError_t gpuMemcpy(void *dst, const void *src, size_t size,
                            cudaMemcpyKind kind) {
  return cudaMemcpy(dst, src, size, kind);
}

inline gpuError_t gpuStreamCreate(gpuStream_t *pStream) {
  return cudaStreamCreate(pStream);
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

inline const char *gpuGetErrorString(gpuError_t error) {
  return cudaGetErrorString(error);
}

#elif defined(OpenPFC_ENABLE_HIP)

// Include HIP headers when HIP is enabled
#include <hip/hip_runtime.h>

#define GPU_CHECK(call)                                                             \
  do {                                                                              \
    hipError_t error = call;                                                        \
    if (error != hipSuccess) {                                                      \
      throw std::runtime_error(std::string("GPU error: ") +                         \
                               hipGetErrorString(error));                           \
    }                                                                               \
  } while (0)

// Type mappings
using gpuStream_t = hipStream_t;
using gpuEvent_t = hipEvent_t;
using gpuError_t = hipError_t;
using gpuMemcpyKind = hipMemcpyKind;

// Function mappings
static constexpr auto gpuSuccess = hipSuccess;
static constexpr auto gpuErrorMemoryAllocation = hipErrorMemoryAllocation;
static constexpr auto gpuMemcpyHostToDevice = hipMemcpyHostToDevice;
static constexpr auto gpuMemcpyDeviceToHost = hipMemcpyDeviceToHost;
static constexpr auto gpuMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
static constexpr auto gpuMemcpyDefault = hipMemcpyDefault;

// Function wrappers
inline gpuError_t gpuMalloc(void **ptr, size_t size) { return hipMalloc(ptr, size); }

inline gpuError_t gpuFree(void *ptr) { return hipFree(ptr); }

inline gpuError_t gpuMemcpyAsync(void *dst, const void *src, size_t size,
                                 hipMemcpyKind kind, gpuStream_t stream) {
  return hipMemcpyAsync(dst, src, size, kind, stream);
}

inline gpuError_t gpuMemcpy(void *dst, const void *src, size_t size,
                            hipMemcpyKind kind) {
  return hipMemcpy(dst, src, size, kind);
}

inline gpuError_t gpuStreamCreate(gpuStream_t *pStream) {
  return hipStreamCreate(pStream);
}

inline gpuError_t gpuStreamDestroy(gpuStream_t stream) {
  return hipStreamDestroy(stream);
}

inline gpuError_t gpuStreamSynchronize(gpuStream_t stream) {
  return hipStreamSynchronize(stream);
}

inline gpuError_t gpuEventCreate(gpuEvent_t *event) { return hipEventCreate(event); }

inline gpuError_t gpuEventDestroy(gpuEvent_t event) {
  return hipEventDestroy(event);
}

inline gpuError_t gpuEventRecord(gpuEvent_t event, gpuStream_t stream) {
  return hipEventRecord(event, stream);
}

inline gpuError_t gpuEventSynchronize(gpuEvent_t event) {
  return hipEventSynchronize(event);
}

inline gpuError_t gpuDeviceSynchronize() { return hipDeviceSynchronize(); }

inline const char *gpuGetErrorString(gpuError_t error) {
  return hipGetErrorString(error);
}

#else

// Error when no GPU backend is enabled
#error                                                                              \
    "GPU API requires either OpenPFC_ENABLE_CUDA or OpenPFC_ENABLE_HIP to be defined"

#endif

// Backend detection for HIPCC compilation
#if defined(__HIPCC__)
#define OPENPFC_HIPCC Compilation
#else
#define OPENPFC_HIPCC 0
#endif

// Launch macros for kernel invocation (common to both backends)
#define GPU_LAUNCH_KERNEL(kernel, grid, block, args, stream)                        \
  do {                                                                              \
    kernel<<<grid, block, 0, stream>>> args;                                        \
    GPU_CHECK(cudaGetLastError());                                                  \
  } while (0)