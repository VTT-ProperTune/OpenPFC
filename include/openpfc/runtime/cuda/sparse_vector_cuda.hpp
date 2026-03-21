// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file sparse_vector_cuda.hpp
 * @brief CudaTag support for SparseVector copy-to-device (runtime/cuda)
 *
 * Include when constructing SparseVector<backend::CudaTag, T> with indices/data.
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA)

#include <cuda_runtime.h>
#include <openpfc/kernel/decomposition/sparse_vector.hpp>
#include <openpfc/runtime/cuda/backend_tags_cuda.hpp>
#include <openpfc/runtime/cuda/databuffer_cuda.hpp>
#include <stdexcept>

namespace pfc {
namespace core {
namespace detail {

template <>
inline void copy_indices_to_device_impl<backend::CudaTag>(
    DataBuffer<backend::CudaTag, size_t> &buf, size_t n,
    const std::vector<size_t> &host_indices) {
  if (n == 0) return;
  cudaError_t err = cudaMemcpy(buf.data(), host_indices.data(), n * sizeof(size_t),
                               cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA copy failed: " +
                             std::string(cudaGetErrorString(err)));
  }
}

template <>
inline void copy_data_to_device_impl<backend::CudaTag, double>(
    DataBuffer<backend::CudaTag, double> &buf, size_t n,
    const std::vector<double> &host_data) {
  if (n == 0) return;
  cudaError_t err = cudaMemcpy(buf.data(), host_data.data(), n * sizeof(double),
                               cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA copy failed: " +
                             std::string(cudaGetErrorString(err)));
  }
}

template <>
inline void copy_data_to_device_impl<backend::CudaTag, float>(
    DataBuffer<backend::CudaTag, float> &buf, size_t n,
    const std::vector<float> &host_data) {
  if (n == 0) return;
  cudaError_t err = cudaMemcpy(buf.data(), host_data.data(), n * sizeof(float),
                               cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA copy failed: " +
                             std::string(cudaGetErrorString(err)));
  }
}

} // namespace detail
} // namespace core
} // namespace pfc

#endif // OpenPFC_ENABLE_CUDA
