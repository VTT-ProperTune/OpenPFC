// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file memory_traits.hpp
 * @brief Type traits for backend metadata and capabilities
 *
 * @details
 * This header provides compile-time type traits that describe the capabilities
 * of different backends. These traits can be used for conditional compilation
 * and static assertions to ensure correct usage of backend-specific features.
 *
 * @code
 * #include <openpfc/core/memory_traits.hpp>
 *
 * // Check if backend supports host access
 * if constexpr (pfc::core::backend_traits<pfc::backend::CpuTag>::has_host_access) {
 *     // Can use operator[]
 *     buffer[0] = 1.0;
 * }
 *
 * // Check if backend requires transfers
 * if constexpr (pfc::core::backend_traits<pfc::backend::CudaTag>::requires_transfer)
 * {
 *     // Must use copy_from_host/to_host
 *     buffer.copy_from_host(host_data);
 * }
 * @endcode
 *
 * @see core/backend_tags.hpp for backend tag definitions
 * @see core/databuffer.hpp for backend-specific implementations
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#pragma once

#include <openpfc/core/backend_tags.hpp>

namespace pfc {
namespace core {

/**
 * @brief Type traits for backend capabilities
 *
 * Provides compile-time information about backend capabilities:
 * - `has_host_access`: Can access memory directly from host (operator[])
 * - `has_device_access`: Memory lives on device (GPU)
 * - `requires_transfer`: Requires explicit CPU-GPU transfers
 *
 * @tparam BackendTag Backend tag to query
 */
template <typename BackendTag> struct backend_traits;

/**
 * @brief CPU backend traits
 *
 * CPU backend has host access, no device access, no transfers needed.
 */
template <> struct backend_traits<backend::CpuTag> {
  /// CPU memory can be accessed directly from host
  static constexpr bool has_host_access = true;
  /// CPU memory is not on device
  static constexpr bool has_device_access = false;
  /// No transfers needed (memory is already on host)
  static constexpr bool requires_transfer = false;
};

#if defined(OpenPFC_ENABLE_CUDA)
/**
 * @brief CUDA backend traits
 *
 * CUDA backend has device access, requires transfers, no host access.
 */
template <> struct backend_traits<backend::CudaTag> {
  /// CUDA memory cannot be accessed directly from host
  static constexpr bool has_host_access = false;
  /// CUDA memory lives on device
  static constexpr bool has_device_access = true;
  /// Requires explicit CPU-GPU transfers
  static constexpr bool requires_transfer = true;
};
#endif

#if defined(OpenPFC_ENABLE_HIP)
/**
 * @brief HIP backend traits
 *
 * HIP backend has device access, requires transfers, no host access.
 */
template <> struct backend_traits<backend::HipTag> {
  /// HIP memory cannot be accessed directly from host
  static constexpr bool has_host_access = false;
  /// HIP memory lives on device
  static constexpr bool has_device_access = true;
  /// Requires explicit CPU-GPU transfers
  static constexpr bool requires_transfer = true;
};
#endif

} // namespace core
} // namespace pfc
