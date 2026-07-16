// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file sparse_vector_ops.hip
 * @brief HIP kernels for SparseVector gather and scatter
 */

#if defined(OpenPFC_ENABLE_HIP)

#include <hip/hip_runtime.h>
#include <openpfc/runtime/hip/sparse_vector_ops_hip.hpp>
#include <stdexcept>
#include <string>

#ifdef OpenPFC_ENABLE_GPU_AUTOTUNING
#include "openpfc/runtime/common/gpu_autotune.hpp"
#endif

namespace pfc {
namespace core {

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

void gather_hip_impl(size_t n, const size_t *indices, double *data,
                      const double *source, size_t source_size) {
  if (n == 0) {
    return;
  }
#ifdef OpenPFC_ENABLE_GPU_AUTOTUNING
  auto config = pfc::gpu::AutoTuner::instance().get_config("gather", n);
  int threads_per_block = config.block_size_x;
#else
  int threads_per_block = 256;
#endif
  const int blocks =
      (static_cast<int>(n) + threads_per_block - 1) / threads_per_block;
  gather_kernel<<<blocks, threads_per_block>>>(n, indices, data, source);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    throw std::runtime_error("HIP gather kernel failed: " +
                             std::string(hipGetErrorString(err)));
  }
  (void)source_size;
}

void scatter_hip_impl(size_t n, const size_t *indices, const double *data,
                       double *dest, size_t dest_size) {
  if (n == 0) {
    return;
  }
#ifdef OpenPFC_ENABLE_GPU_AUTOTUNING
  auto config = pfc::gpu::AutoTuner::instance().get_config("scatter", n);
  int threads_per_block = config.block_size_x;
#else
  int threads_per_block = 256;
#endif
  const int blocks =
      (static_cast<int>(n) + threads_per_block - 1) / threads_per_block;
  scatter_kernel<<<blocks, threads_per_block>>>(n, indices, data, dest);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    throw std::runtime_error("HIP scatter kernel failed: " +
                             std::string(hipGetErrorString(err)));
  }
  (void)dest_size;
}

} // namespace core
} // namespace pfc

#endif
