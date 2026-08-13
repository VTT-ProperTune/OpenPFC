// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file for_each_interior_device.hpp
 * @brief HIP `for_each_interior_device` (re-export of the M3 GPU source).
 *
 * @see runtime/gpu/for_each_interior_device_gpu.hpp
 */

#pragma once

#include <openpfc/runtime/gpu/for_each_interior_device_gpu.hpp>

#if defined(OpenPFC_ENABLE_HIP)

namespace pfc::sim::hip {

using ::pfc::sim::gpu::DevicePtrPack2;
using ::pfc::sim::gpu::DevicePtrPack3;
using ::pfc::sim::gpu::DevicePtrPack4;
using ::pfc::sim::gpu::DeviceInc2;
using ::pfc::sim::gpu::DeviceInc3;
using ::pfc::sim::gpu::DeviceInc4;
using ::pfc::sim::gpu::make_device_ptr_pack;
using ::pfc::sim::gpu::is_device_ptr_pack;
using ::pfc::sim::gpu::for_each_interior_device;

namespace detail {

using ::pfc::sim::gpu::detail::scatter_device;
using ::pfc::sim::gpu::detail::for_each_interior_device_kernel;
using ::pfc::sim::gpu::detail::for_each_interior_device_kernel_multi;
using ::pfc::sim::gpu::detail::for_each_interior_grid;
using ::pfc::sim::gpu::detail::for_each_interior_block;

} // namespace detail

} // namespace pfc::sim::hip

#endif // OpenPFC_ENABLE_HIP
