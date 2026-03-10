// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file layout.hpp
 * @brief Kokkos-compatible layout types and stride computation
 *
 * @details
 * Layout types determine how multi-dimensional indices map to linear storage.
 * Names and semantics match Kokkos.
 *
 * - LayoutRight: C-style; rightmost index contiguous (stride 1 in last dim).
 * - LayoutLeft: Fortran-style; leftmost index contiguous (stride 1 in first dim).
 *
 * @see view.hpp for View template using these layouts
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#pragma once

#include <array>
#include <cstddef>

namespace pfc {

/**
 * @brief LayoutRight: C-style layout
 *
 * Strides increase from right to left. Last dimension has stride 1.
 * Corresponds to C/C++ multi-dimensional arrays (e.g. a[i][j][k]).
 */
struct LayoutRight {};

/**
 * @brief LayoutLeft: Fortran-style layout
 *
 * Strides increase from left to right. First dimension has stride 1.
 */
struct LayoutLeft {};

namespace detail {

template <std::size_t Rank>
std::array<std::size_t, Rank>
strides_from_extents_layout_right(const std::array<std::size_t, Rank> &extents) {
  std::array<std::size_t, Rank> strides{};
  if (Rank == 0) return strides;
  strides[Rank - 1] = 1;
  for (std::size_t r = Rank - 1; r > 0; --r) {
    strides[r - 1] = strides[r] * extents[r];
  }
  return strides;
}

template <std::size_t Rank>
std::array<std::size_t, Rank>
strides_from_extents_layout_left(const std::array<std::size_t, Rank> &extents) {
  std::array<std::size_t, Rank> strides{};
  if (Rank == 0) return strides;
  strides[0] = 1;
  for (std::size_t r = 1; r < Rank; ++r) {
    strides[r] = strides[r - 1] * extents[r - 1];
  }
  return strides;
}

template <std::size_t Rank>
std::size_t span_from_extents(const std::array<std::size_t, Rank> &extents) {
  std::size_t s = 1;
  for (std::size_t r = 0; r < Rank; ++r) s *= extents[r];
  return s;
}

} // namespace detail

/**
 * @brief Compute strides for the given layout and extents
 *
 * @tparam Layout LayoutRight or LayoutLeft
 * @tparam Rank Number of dimensions
 * @param extents Extent per dimension
 * @return Array of strides (elements between consecutive indices in each dim)
 */
template <typename Layout, std::size_t Rank>
std::array<std::size_t, Rank>
strides_from_extents(const std::array<std::size_t, Rank> &extents);

template <std::size_t Rank>
std::array<std::size_t, Rank>
strides_from_extents(const std::array<std::size_t, Rank> &extents, LayoutRight) {
  return detail::strides_from_extents_layout_right(extents);
}

template <std::size_t Rank>
std::array<std::size_t, Rank>
strides_from_extents(const std::array<std::size_t, Rank> &extents, LayoutLeft) {
  return detail::strides_from_extents_layout_left(extents);
}

template <typename Layout, std::size_t Rank>
std::array<std::size_t, Rank>
strides_from_extents(const std::array<std::size_t, Rank> &extents) {
  return strides_from_extents(extents, Layout{});
}

/**
 * @brief Compute linear index from multi-dimensional indices
 *
 * @tparam Rank Number of dimensions
 * @param indices Multi-dimensional indices
 * @param strides Strides from strides_from_extents
 * @return Linear index
 */
template <std::size_t Rank>
std::size_t linear_index_from_strides(const std::array<std::size_t, Rank> &indices,
                                      const std::array<std::size_t, Rank> &strides) {
  std::size_t idx = 0;
  for (std::size_t r = 0; r < Rank; ++r) idx += indices[r] * strides[r];
  return idx;
}

/**
 * @brief Compute linear index from indices and extents (layout-aware)
 */
template <typename Layout, std::size_t Rank>
std::size_t linear_index(const std::array<std::size_t, Rank> &indices,
                         const std::array<std::size_t, Rank> &extents) {
  auto strides = strides_from_extents<Layout>(extents);
  return linear_index_from_strides(indices, strides);
}

} // namespace pfc
