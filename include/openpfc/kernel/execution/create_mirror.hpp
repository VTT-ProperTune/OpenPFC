// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file create_mirror.hpp
 * @brief Kokkos-compatible create_mirror and create_mirror_view
 *
 * @details
 * create_mirror: allocate a new View in host memory with same layout/extents.
 * create_mirror_view: return a host-accessible View; if the view is already
 * in host memory, returns a View wrapping the same data (shallow); otherwise
 * allocates a mirror. Names and semantics match Kokkos.
 *
 * @see view.hpp for View
 * @see deep_copy.hpp for copying between views
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#pragma once

#include <openpfc/kernel/execution/view.hpp>

namespace pfc {

/**
 * @brief Create a new View in host memory with same layout and extents
 * (Kokkos-compatible). Always allocates; independent of source.
 */
template <typename T, typename Layout, typename MemorySpace>
View<T, 1, Layout, HostSpace>
create_mirror(const View<T, 1, Layout, MemorySpace> &src) {
  return View<T, 1, Layout, HostSpace>("mirror", src.extent(0));
}

template <typename T, typename Layout, typename MemorySpace>
View<T, 2, Layout, HostSpace>
create_mirror(const View<T, 2, Layout, MemorySpace> &src) {
  return View<T, 2, Layout, HostSpace>("mirror", src.extent(0), src.extent(1));
}

template <typename T, typename Layout, typename MemorySpace>
View<T, 3, Layout, HostSpace>
create_mirror(const View<T, 3, Layout, MemorySpace> &src) {
  return View<T, 3, Layout, HostSpace>("mirror", src.extent(0), src.extent(1),
                                       src.extent(2));
}

/**
 * @brief Create host-accessible View (Kokkos-compatible).
 * If src is already in HostSpace, returns a View that wraps the same data
 * (unmanaged view of src.data() with same extents). Otherwise allocates
 * a new mirror (same as create_mirror).
 */
template <typename T, typename Layout, typename MemorySpace>
View<T, 1, Layout, HostSpace>
create_mirror_view(const View<T, 1, Layout, MemorySpace> &src) {
  if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
    return View<T, 1, Layout, HostSpace>(const_cast<T *>(src.data()), src.extent(0));
  }
  return create_mirror(src);
}

template <typename T, typename Layout, typename MemorySpace>
View<T, 2, Layout, HostSpace>
create_mirror_view(const View<T, 2, Layout, MemorySpace> &src) {
  if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
    return View<T, 2, Layout, HostSpace>(const_cast<T *>(src.data()), src.extent(0),
                                         src.extent(1));
  }
  return create_mirror(src);
}

template <typename T, typename Layout, typename MemorySpace>
View<T, 3, Layout, HostSpace>
create_mirror_view(const View<T, 3, Layout, MemorySpace> &src) {
  if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
    return View<T, 3, Layout, HostSpace>(const_cast<T *>(src.data()), src.extent(0),
                                         src.extent(1), src.extent(2));
  }
  return create_mirror(src);
}

} // namespace pfc
