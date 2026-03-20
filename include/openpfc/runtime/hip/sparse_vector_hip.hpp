// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file sparse_vector_hip.hpp
 * @brief HipTag support for SparseVector copy-to-device (runtime/hip)
 *
 * Include when constructing SparseVector<backend::HipTag, T> with indices/data.
 */

#pragma once

#if defined(OpenPFC_ENABLE_HIP)

#include <hip/hip_runtime.h>
#include <openpfc/kernel/decomposition/sparse_vector.hpp>
#include <openpfc/runtime/hip/backend_tags_hip.hpp>
#include <openpfc/runtime/hip/databuffer_hip.hpp>
#include <stdexcept>

namespace pfc {
namespace core {
namespace detail {

template <>
inline void copy_indices_to_device_impl<backend::HipTag>(
    DataBuffer<backend::HipTag, size_t> &buf, size_t n,
    const std::vector<size_t> &host_indices) {
  if (n == 0) return;
  hipError_t err = hipMemcpy(buf.data(), host_indices.data(), n * sizeof(size_t),
                             hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    throw std::runtime_error("HIP copy failed");
  }
}

template <>
inline void copy_data_to_device_impl<backend::HipTag, double>(
    DataBuffer<backend::HipTag, double> &buf, size_t n,
    const std::vector<double> &host_data) {
  if (n == 0) return;
  hipError_t err = hipMemcpy(buf.data(), host_data.data(), n * sizeof(double),
                             hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    throw std::runtime_error("HIP copy failed");
  }
}

template <>
inline void copy_data_to_device_impl<backend::HipTag, float>(
    DataBuffer<backend::HipTag, float> &buf, size_t n,
    const std::vector<float> &host_data) {
  if (n == 0) return;
  hipError_t err = hipMemcpy(buf.data(), host_data.data(), n * sizeof(float),
                             hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    throw std::runtime_error("HIP copy failed");
  }
}

} // namespace detail
} // namespace core
} // namespace pfc

#endif // OpenPFC_ENABLE_HIP
