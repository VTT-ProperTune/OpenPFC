// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file sparse_vector_gpu.hpp
 * @brief Single-source GPU SparseVector copy-to-device for CUDA and HIP (M3).
 *
 * Specializes `copy_indices_to_device_impl` / `copy_data_to_device_impl` for
 * `CudaTag` and/or `HipTag`. Vendor headers `sparse_vector_cuda.hpp` /
 * `sparse_vector_hip.hpp` are thin includes of this file so existing call
 * sites keep compiling.
 *
 * Per-tag memcpy calls the native runtime (not `gpu_api.hpp`) so a CUDA+HIP
 * co-enabled translation unit can own both specializations. Failures throw
 * via `cuda_check` / `hip_check` as `"CUDA copy failed: …"` /
 * `"HIP copy failed: …"`.
 *
 * @see kernel/decomposition/sparse_vector.hpp
 * @see runtime/gpu/databuffer_gpu.hpp
 * @see runtime/gpu/gpu_check.hpp
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <cstddef>
#include <vector>

#include <openpfc/kernel/decomposition/sparse_vector.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/gpu_check.hpp>

#if defined(OpenPFC_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif
#if defined(OpenPFC_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace pfc::core::detail {

#if defined(OpenPFC_ENABLE_CUDA)
struct CudaH2D {
  using tag = backend::CudaTag;
  static void memcpy_h2d(void *dst, const void *src, std::size_t bytes) {
    pfc::cuda::detail::cuda_check(
        cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "CUDA copy failed");
  }
};
#endif

#if defined(OpenPFC_ENABLE_HIP)
struct HipH2D {
  using tag = backend::HipTag;
  static void memcpy_h2d(void *dst, const void *src, std::size_t bytes) {
    pfc::hip::detail::hip_check(
        hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice), "HIP copy failed");
  }
};
#endif

template <typename Ops>
void gpu_copy_h2d(void *dst, const void *src, std::size_t n, std::size_t elem) {
  if (n == 0) {
    return;
  }
  Ops::memcpy_h2d(dst, src, n * elem);
}

#if defined(OpenPFC_ENABLE_CUDA)
template <>
inline void copy_indices_to_device_impl<backend::CudaTag>(
    DataBuffer<backend::CudaTag, size_t> &buf, size_t n,
    const std::vector<size_t> &host_indices) {
  gpu_copy_h2d<CudaH2D>(buf.data(), host_indices.data(), n, sizeof(size_t));
}

template <>
inline void copy_data_to_device_impl<backend::CudaTag, double>(
    DataBuffer<backend::CudaTag, double> &buf, size_t n,
    const std::vector<double> &host_data) {
  gpu_copy_h2d<CudaH2D>(buf.data(), host_data.data(), n, sizeof(double));
}

template <>
inline void copy_data_to_device_impl<backend::CudaTag, float>(
    DataBuffer<backend::CudaTag, float> &buf, size_t n,
    const std::vector<float> &host_data) {
  gpu_copy_h2d<CudaH2D>(buf.data(), host_data.data(), n, sizeof(float));
}
#endif

#if defined(OpenPFC_ENABLE_HIP)
template <>
inline void copy_indices_to_device_impl<backend::HipTag>(
    DataBuffer<backend::HipTag, size_t> &buf, size_t n,
    const std::vector<size_t> &host_indices) {
  gpu_copy_h2d<HipH2D>(buf.data(), host_indices.data(), n, sizeof(size_t));
}

template <>
inline void copy_data_to_device_impl<backend::HipTag, double>(
    DataBuffer<backend::HipTag, double> &buf, size_t n,
    const std::vector<double> &host_data) {
  gpu_copy_h2d<HipH2D>(buf.data(), host_data.data(), n, sizeof(double));
}

template <>
inline void copy_data_to_device_impl<backend::HipTag, float>(
    DataBuffer<backend::HipTag, float> &buf, size_t n,
    const std::vector<float> &host_data) {
  gpu_copy_h2d<HipH2D>(buf.data(), host_data.data(), n, sizeof(float));
}
#endif

} // namespace pfc::core::detail

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
