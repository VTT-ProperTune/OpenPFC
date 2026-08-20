// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file fd_gradient_device.hpp
 * @brief CUDA FD device gradient evaluator (re-export of the M3 GPU source).
 *
 * @see runtime/gpu/fd_gradient_device_gpu.hpp
 */

#pragma once

#include <openpfc/runtime/gpu/fd_gradient_device_gpu.hpp>

#if defined(OpenPFC_ENABLE_CUDA)

namespace pfc::cuda {

using ::pfc::gpu::kFDDeviceMaxHw1;
using ::pfc::gpu::kFDDeviceMaxHw2;
using ::pfc::gpu::kMaxCompositeFields;
using ::pfc::gpu::FDGradientDevicePOD;
using ::pfc::gpu::CompositeGradientDevicePOD;
using ::pfc::gpu::evaluate_fd_grad;
using ::pfc::gpu::evaluate_fd_grad_composite;
using ::pfc::gpu::FDGradientDevice;
using ::pfc::gpu::CompositeGradientDevice;
using ::pfc::gpu::create_composite_device;
using ::pfc::gpu::create;

} // namespace pfc::cuda

#endif // OpenPFC_ENABLE_CUDA
