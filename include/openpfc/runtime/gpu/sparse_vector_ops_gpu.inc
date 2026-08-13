// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file sparse_vector_ops_gpu.inc
 * @brief Single-source GPU gather/scatter kernels (included from vendor .cu/.hip).
 *
 * Do not include this from a host TU. Error strings keep the vendor kind
 * (`"CUDA gather kernel failed: …"` / `"HIP gather kernel failed: …"`).
 * OOB messages match CPU (`gather: index out of bounds` /
 * `scatter: index out of bounds`).
 *
 * @see runtime/gpu/sparse_vector_ops_gpu.hpp
 */

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/runtime/gpu/gpu_api.hpp>

#ifdef OpenPFC_ENABLE_GPU_AUTOTUNING
#include "openpfc/runtime/common/gpu_autotune.hpp"
#endif

namespace pfc::core {

#if defined(OPENPFC_GPU_API_CUDA)
constexpr const char *kGpuKind = "CUDA";
#elif defined(OPENPFC_GPU_API_HIP)
constexpr const char *kGpuKind = "HIP";
#endif

__global__ void gather_kernel(size_t n, const size_t *indices, double *data,
                              const double *source) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    size_t idx = indices[i];
    data[i] = source[idx];
  }
}

__global__ void scatter_kernel(size_t n, const size_t *indices, const double *data,
                               double *dest) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    size_t idx = indices[i];
    dest[idx] = data[i];
  }
}

namespace {

void check_indices_host(size_t n, const size_t *indices, size_t length,
                        const char *oob_message) {
  std::vector<size_t> host_idx(n);
  ::pfc::gpuError_t err = ::pfc::gpuMemcpy(host_idx.data(), indices,
                                           n * sizeof(size_t),
                                           ::pfc::gpuMemcpyDeviceToHost);
  if (err != ::pfc::gpuSuccess) {
    throw std::runtime_error(std::string(::pfc::gpuGetErrorString(err)));
  }
  for (size_t i = 0; i < n; ++i) {
    if (host_idx[i] >= length) {
      throw std::runtime_error(oob_message);
    }
  }
}

int gather_scatter_threads(size_t n, const char *kernel_name) {
#ifdef OpenPFC_ENABLE_GPU_AUTOTUNING
  auto config = pfc::gpu::AutoTuner::instance().get_config(kernel_name, n);
  return config.block_size_x;
#else
  (void)n;
  (void)kernel_name;
  return 256;
#endif
}

void launch_gather(size_t n, const size_t *indices, double *data,
                   const double *source, size_t source_size) {
  if (n == 0) {
    return;
  }
  check_indices_host(n, indices, source_size, "gather: index out of bounds");
  const int threads_per_block = gather_scatter_threads(n, "gather");
  const int blocks =
      (static_cast<int>(n) + threads_per_block - 1) / threads_per_block;
  gather_kernel<<<blocks, threads_per_block>>>(n, indices, data, source);
  ::pfc::gpuError_t err = ::pfc::gpuGetLastError();
  if (err != ::pfc::gpuSuccess) {
    throw std::runtime_error(std::string(kGpuKind) + " gather kernel failed: " +
                             ::pfc::gpuGetErrorString(err));
  }
  err = ::pfc::gpuDeviceSynchronize();
  if (err != ::pfc::gpuSuccess) {
    throw std::runtime_error(std::string(kGpuKind) +
                             " gather kernel sync failed: " +
                             ::pfc::gpuGetErrorString(err));
  }
}

void launch_scatter(size_t n, const size_t *indices, const double *data,
                    double *dest, size_t dest_size) {
  if (n == 0) {
    return;
  }
  check_indices_host(n, indices, dest_size, "scatter: index out of bounds");
  const int threads_per_block = gather_scatter_threads(n, "scatter");
  const int blocks =
      (static_cast<int>(n) + threads_per_block - 1) / threads_per_block;
  scatter_kernel<<<blocks, threads_per_block>>>(n, indices, data, dest);
  ::pfc::gpuError_t err = ::pfc::gpuGetLastError();
  if (err != ::pfc::gpuSuccess) {
    throw std::runtime_error(std::string(kGpuKind) + " scatter kernel failed: " +
                             ::pfc::gpuGetErrorString(err));
  }
  err = ::pfc::gpuDeviceSynchronize();
  if (err != ::pfc::gpuSuccess) {
    throw std::runtime_error(std::string(kGpuKind) +
                             " scatter kernel sync failed: " +
                             ::pfc::gpuGetErrorString(err));
  }
}

} // namespace

#if defined(OpenPFC_ENABLE_CUDA) && defined(OPENPFC_GPU_API_CUDA)
void gather_cuda_impl(size_t n, const size_t *indices, double *data,
                      const double *source, size_t source_size) {
  launch_gather(n, indices, data, source, source_size);
}

void scatter_cuda_impl(size_t n, const size_t *indices, const double *data,
                       double *dest, size_t dest_size) {
  launch_scatter(n, indices, data, dest, dest_size);
}
#endif

#if defined(OpenPFC_ENABLE_HIP) && defined(OPENPFC_GPU_API_HIP)
void gather_hip_impl(size_t n, const size_t *indices, double *data,
                     const double *source, size_t source_size) {
  launch_gather(n, indices, data, source, source_size);
}

void scatter_hip_impl(size_t n, const size_t *indices, const double *data,
                      double *dest, size_t dest_size) {
  launch_scatter(n, indices, data, dest, dest_size);
}
#endif

} // namespace pfc::core
