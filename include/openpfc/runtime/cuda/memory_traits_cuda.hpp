// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file memory_traits_cuda.hpp
 * @brief CUDA backend traits (runtime/cuda only)
 *
 * @see kernel/execution/memory_traits.hpp for CpuTag and interface
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA)

#include <openpfc/kernel/execution/memory_traits.hpp>
#include <openpfc/runtime/cuda/backend_tags_cuda.hpp>

namespace pfc {
namespace core {

template <> struct backend_traits<backend::CudaTag> {
  static constexpr bool has_host_access = false;
  static constexpr bool has_device_access = true;
  static constexpr bool requires_transfer = true;
};

} // namespace core
} // namespace pfc

#endif // OpenPFC_ENABLE_CUDA
