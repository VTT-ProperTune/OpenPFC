// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file deep_copy_cuda.hpp
 * @brief CUDA device-to-device deep_copy (runtime/cuda)
 *
 * Include after openpfc/kernel/execution/deep_copy.hpp when copying between
 * CudaSpace Views.
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA)

#include <cuda_runtime.h>
#include <openpfc/kernel/execution/deep_copy.hpp>
#include <openpfc/kernel/execution/view.hpp>
#include <openpfc/runtime/cuda/memory_space_cuda.hpp>
#include <stdexcept>

namespace pfc {
namespace detail {

template <typename T, std::size_t Rank, typename L1, typename L2>
void deep_copy_view_to_view_impl(View<T, Rank, L1, CudaSpace> &dst,
                                 const View<T, Rank, L2, CudaSpace> &src) {
  const std::size_t n = dst.size();
  if (src.size() != n) {
    throw std::runtime_error("deep_copy: View size mismatch");
  }
  if (n == 0) return;
  cudaError_t err =
      cudaMemcpy(dst.data(), src.data(), n * sizeof(T), cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess) {
    throw std::runtime_error("deep_copy: CUDA device-to-device failed");
  }
}

} // namespace detail
} // namespace pfc

#endif // OpenPFC_ENABLE_CUDA
