// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file memory_traits_hip.hpp
 * @brief HIP backend traits (runtime/hip only)
 *
 * @see kernel/execution/memory_traits.hpp for CpuTag and interface
 */

#pragma once

#if defined(OpenPFC_ENABLE_HIP)

#include <openpfc/kernel/execution/memory_traits.hpp>
#include <openpfc/runtime/hip/backend_tags_hip.hpp>

namespace pfc {
namespace core {

template <> struct backend_traits<backend::HipTag> {
  static constexpr bool has_host_access = false;
  static constexpr bool has_device_access = true;
  static constexpr bool requires_transfer = true;
};

} // namespace core
} // namespace pfc

#endif // OpenPFC_ENABLE_HIP
