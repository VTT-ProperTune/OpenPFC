// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file memory_space_hip.hpp
 * @brief HIP memory space tag (runtime/hip only)
 *
 * @see kernel/execution/memory_space.hpp for HostSpace
 * @see runtime/cuda/memory_space_cuda.hpp for CudaSpace
 */

#pragma once

#if defined(OpenPFC_ENABLE_HIP)

#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/runtime/hip/backend_tags_hip.hpp>

namespace pfc {

struct HipSpace {};

template <> struct memory_space_to_backend<HipSpace> {
  using type = backend::HipTag;
};

} // namespace pfc

#endif // OpenPFC_ENABLE_HIP
