// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file memory_space_cuda.hpp
 * @brief CUDA memory space tag (runtime/cuda only)
 *
 * @see kernel/execution/memory_space.hpp for HostSpace
 * @see runtime/hip/memory_space_hip.hpp for HipSpace
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA)

#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/runtime/cuda/backend_tags_cuda.hpp>

namespace pfc {

struct CudaSpace {};

template <> struct memory_space_to_backend<CudaSpace> {
  using type = backend::CudaTag;
};

} // namespace pfc

#endif // OpenPFC_ENABLE_CUDA
