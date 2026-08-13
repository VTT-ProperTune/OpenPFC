// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file fill_gpu.hpp
 * @brief Device scalar fill kernels for CUDA and HIP (M3).
 *
 * Compiled from `src/openpfc/runtime/gpu/fill.cu` / `fill.hip`. Used by
 * `deep_copy(buffer, scalar)` (and remaining View fill overloads) so device
 * fills do not stage a host vector.
 *
 * @see runtime/gpu/deep_copy_gpu.hpp
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <cstddef>

namespace pfc {

#if defined(OpenPFC_ENABLE_CUDA)
void fill_cuda_impl(double *ptr, std::size_t n, double value);
void fill_cuda_impl(float *ptr, std::size_t n, float value);
#endif

#if defined(OpenPFC_ENABLE_HIP)
void fill_hip_impl(double *ptr, std::size_t n, double value);
void fill_hip_impl(float *ptr, std::size_t n, float value);
#endif

} // namespace pfc

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
