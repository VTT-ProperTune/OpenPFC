// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file sparse_vector_ops.cu
 * @brief CUDA kernels for SparseVector gather and scatter
 */

#if defined(OpenPFC_ENABLE_CUDA)

#include <cuda_runtime.h>
#include <openpfc/runtime/cuda/sparse_vector_ops_cuda.hpp>
#include <stdexcept>
#include <string>
#include <vector>

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

namespace {

void check_indices_host(size_t n, const size_t *indices, size_t length,
                        const char *oob_message) {
  std::vector<size_t> host_idx(n);
  cudaError_t err =
      cudaMemcpy(host_idx.data(), indices, n * sizeof(size_t),
                 cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(cudaGetErrorString(err)));
  }
  for (size_t i = 0; i < n; ++i) {
    if (host_idx[i] >= length) {
      throw std::runtime_error(oob_message);
    }
  }
}

} // namespace

void gather_cuda_impl(size_t n, const size_t *indices, double *data,
                      const double *source, size_t source_size) {
  if (n == 0) {
    return;
  }
  check_indices_host(n, indices, source_size, "gather: index out of bounds");
#ifdef OpenPFC_ENABLE_GPU_AUTOTUNING
  auto config = pfc::gpu::AutoTuner::instance().get_config("gather", n);
  int threads_per_block = config.block_size_x;
#else
  int threads_per_block = 256;
#endif
  const int blocks =
      (static_cast<int>(n) + threads_per_block - 1) / threads_per_block;
  gather_kernel<<<blocks, threads_per_block>>>(n, indices, data, source);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA gather kernel failed: " +
                             std::string(cudaGetErrorString(err)));
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA gather kernel sync failed: " +
                             std::string(cudaGetErrorString(err)));
  }
}

void scatter_cuda_impl(size_t n, const size_t *indices, const double *data,
                       double *dest, size_t dest_size) {
  if (n == 0) {
    return;
  }
  check_indices_host(n, indices, dest_size, "scatter: index out of bounds");
#ifdef OpenPFC_ENABLE_GPU_AUTOTUNING
  auto config = pfc::gpu::AutoTuner::instance().get_config("scatter", n);
  int threads_per_block = config.block_size_x;
#else
  int threads_per_block = 256;
#endif
  const int blocks =
      (static_cast<int>(n) + threads_per_block - 1) / threads_per_block;
  scatter_kernel<<<blocks, threads_per_block>>>(n, indices, data, dest);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA scatter kernel failed: " +
                             std::string(cudaGetErrorString(err)));
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA scatter kernel sync failed: " +
                             std::string(cudaGetErrorString(err)));
  }
}

} // namespace core
} // namespace pfc

#endif
