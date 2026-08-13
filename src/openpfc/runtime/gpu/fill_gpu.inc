// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file fill_gpu.inc
 * @brief Single-source GPU scalar fill kernel (included from vendor .cu/.hip).
 *
 * Do not include this from a host TU. Error strings keep the vendor kind
 * (`"CUDA fill kernel failed: …"` / `"HIP fill kernel failed: …"`).
 *
 * @see runtime/gpu/fill_gpu.hpp
 */

#include <cstddef>
#include <stdexcept>
#include <string>

#include <openpfc/runtime/gpu/gpu_api.hpp>

namespace pfc {

#if defined(OPENPFC_GPU_API_CUDA)
constexpr const char *kFillGpuKind = "CUDA";
#elif defined(OPENPFC_GPU_API_HIP)
constexpr const char *kFillGpuKind = "HIP";
#endif

template <typename T>
__global__ void fill_kernel(T *ptr, std::size_t n, T value) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    ptr[i] = value;
  }
}

template <typename T> void launch_fill(T *ptr, std::size_t n, T value) {
  if (n == 0) {
    return;
  }
  constexpr int threads_per_block = 256;
  const int blocks =
      static_cast<int>((n + static_cast<std::size_t>(threads_per_block) - 1) /
                       static_cast<std::size_t>(threads_per_block));
  fill_kernel<<<blocks, threads_per_block>>>(ptr, n, value);
  ::pfc::gpuError_t err = ::pfc::gpuGetLastError();
  if (err != ::pfc::gpuSuccess) {
    throw std::runtime_error(std::string(kFillGpuKind) + " fill kernel failed: " +
                             ::pfc::gpuGetErrorString(err));
  }
  err = ::pfc::gpuDeviceSynchronize();
  if (err != ::pfc::gpuSuccess) {
    throw std::runtime_error(std::string(kFillGpuKind) +
                             " fill kernel sync failed: " +
                             ::pfc::gpuGetErrorString(err));
  }
}

#if defined(OpenPFC_ENABLE_CUDA) && defined(OPENPFC_GPU_API_CUDA)
void fill_cuda_impl(double *ptr, std::size_t n, double value) {
  launch_fill(ptr, n, value);
}
void fill_cuda_impl(float *ptr, std::size_t n, float value) {
  launch_fill(ptr, n, value);
}
#endif

#if defined(OpenPFC_ENABLE_HIP) && defined(OPENPFC_GPU_API_HIP)
void fill_hip_impl(double *ptr, std::size_t n, double value) {
  launch_fill(ptr, n, value);
}
void fill_hip_impl(float *ptr, std::size_t n, float value) {
  launch_fill(ptr, n, value);
}
#endif

} // namespace pfc
