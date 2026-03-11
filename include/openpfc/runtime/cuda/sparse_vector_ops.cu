// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
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

void gather_cuda_impl(size_t n, const size_t *indices, double *data,
                      const double *source, size_t source_size) {
  if (n == 0) {
    return;
  }
  const int threads_per_block = 256;
  const int blocks =
      (static_cast<int>(n) + threads_per_block - 1) / threads_per_block;
  gather_kernel<<<blocks, threads_per_block>>>(n, indices, data, source);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA gather kernel failed: " +
                             std::string(cudaGetErrorString(err)));
  }
  (void)source_size;
}

void scatter_cuda_impl(size_t n, const size_t *indices, const double *data,
                       double *dest, size_t dest_size) {
  if (n == 0) {
    return;
  }
  const int threads_per_block = 256;
  const int blocks =
      (static_cast<int>(n) + threads_per_block - 1) / threads_per_block;
  scatter_kernel<<<blocks, threads_per_block>>>(n, indices, data, dest);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA scatter kernel failed: " +
                             std::string(cudaGetErrorString(err)));
  }
  (void)dest_size;
}

} // namespace core
} // namespace pfc

#endif
