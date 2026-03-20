// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file view_hip.hpp
 * @brief HIP memory space execution space mapping (runtime/hip only)
 *
 * Include with view.hpp when using View with HipSpace.
 *
 * @see kernel/execution/view.hpp for View and HostSpace mapping
 * @see runtime/cuda/view_cuda.hpp for CudaSpace
 */

#pragma once

#if defined(OpenPFC_ENABLE_HIP)

#include <openpfc/kernel/execution/view.hpp>
#include <openpfc/runtime/hip/execution_space_hip.hpp>
#include <openpfc/runtime/hip/memory_space_hip.hpp>

namespace pfc {
namespace detail {

template <> struct memory_space_execution_space<HipSpace> {
  using type = HIP;
};

} // namespace detail
} // namespace pfc

#endif // OpenPFC_ENABLE_HIP
