// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file kernels_simple.cu
 * @brief CUDA kernel implementations for simple element-wise operations
 *
 * This file contains CUDA kernel code. It is only compiled when CUDA is enabled.
 */

#if defined(OPENPFC_ENABLE_CUDA)

#include <cuda_runtime.h>
#include <openpfc/gpu/kernels_simple.hpp>
#include <stdexcept>
#include <string>

namespace pfc {
namespace gpu {

// CUDA kernel: Add scalar to each element
__global__ void add_scalar_kernel(double *data, double scalar, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] += scalar;
  }
}

// CUDA kernel: Multiply each element by scalar
__global__ void multiply_scalar_kernel(double *data, double scalar, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] *= scalar;
  }
}

// Host wrapper: Add scalar
void add_scalar(GPUVector<double> &vec, double value) {
  const size_t n = vec.size();
  if (n == 0) {
    return; // Nothing to do
  }

  const int threads_per_block = 256;
  const int blocks =
      (static_cast<int>(n) + threads_per_block - 1) / threads_per_block;

  add_scalar_kernel<<<blocks, threads_per_block>>>(vec.data(), value, n);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA kernel launch failed (add_scalar): " +
                             std::string(cudaGetErrorString(err)));
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA kernel execution failed (add_scalar): " +
                             std::string(cudaGetErrorString(err)));
  }
}

// Host wrapper: Multiply scalar
void multiply_scalar(GPUVector<double> &vec, double value) {
  const size_t n = vec.size();
  if (n == 0) {
    return; // Nothing to do
  }

  const int threads_per_block = 256;
  const int blocks =
      (static_cast<int>(n) + threads_per_block - 1) / threads_per_block;

  multiply_scalar_kernel<<<blocks, threads_per_block>>>(vec.data(), value, n);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA kernel launch failed (multiply_scalar): " +
                             std::string(cudaGetErrorString(err)));
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA kernel execution failed (multiply_scalar): " +
                             std::string(cudaGetErrorString(err)));
  }
}

} // namespace gpu
} // namespace pfc

#endif // OPENPFC_ENABLE_CUDA
