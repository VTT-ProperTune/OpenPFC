// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file deep_copy_gpu.hpp
 * @brief Single-source GPU `deep_copy` scalar fill for CUDA and HIP (M3).
 *
 * `deep_copy(buffer, scalar)` runs a device fill kernel instead of staging a
 * host vector. Vendor headers `deep_copy_cuda.hpp` / `deep_copy_hip.hpp` are
 * thin includes of this file. Device `View` fill and View-to-View device
 * copies are not provided; use `DataBuffer`.
 *
 * @see runtime/gpu/fill_gpu.hpp
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/fill_gpu.hpp>

namespace pfc {

#if defined(OpenPFC_ENABLE_CUDA)
inline void deep_copy(core::DataBuffer<backend::CudaTag, double> &dst, double value) {
  fill_cuda_impl(dst.data(), dst.size(), value);
}
inline void deep_copy(core::DataBuffer<backend::CudaTag, float> &dst, float value) {
  fill_cuda_impl(dst.data(), dst.size(), value);
}
#endif

#if defined(OpenPFC_ENABLE_HIP)
inline void deep_copy(core::DataBuffer<backend::HipTag, double> &dst, double value) {
  fill_hip_impl(dst.data(), dst.size(), value);
}
inline void deep_copy(core::DataBuffer<backend::HipTag, float> &dst, float value) {
  fill_hip_impl(dst.data(), dst.size(), value);
}
#endif

} // namespace pfc

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
