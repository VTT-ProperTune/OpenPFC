// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file view_cuda.hpp
 * @brief CUDA memory space execution space mapping (runtime/cuda only)
 *
 * Include with view.hpp when using View with CudaSpace.
 *
 * @see kernel/execution/view.hpp for View and HostSpace mapping
 * @see runtime/hip/view_hip.hpp for HipSpace
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA)

#include <openpfc/kernel/execution/view.hpp>
#include <openpfc/runtime/cuda/execution_space_cuda.hpp>
#include <openpfc/runtime/cuda/memory_space_cuda.hpp>

namespace pfc {
namespace detail {

template <> struct memory_space_execution_space<CudaSpace> { using type = Cuda; };

} // namespace detail
} // namespace pfc

#endif // OpenPFC_ENABLE_CUDA
