// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file view.hpp
 * @brief Kokkos-compatible multi-dimensional View
 *
 * @details
 * View is a multi-dimensional array with compile-time layout and memory space.
 * API matches Kokkos::View so that switching to Kokkos later requires minimal
 * code changes. Uses DataBuffer internally for allocation and host/device
 * transfer.
 *
 * @see layout.hpp for LayoutRight, LayoutLeft
 * @see memory_space.hpp for HostSpace, CudaSpace
 * @see kernel/execution/databuffer.hpp for underlying storage
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#pragma once

#include <array>
#include <cstddef>
#include <openpfc/kernel/execution/databuffer.hpp>
#include <openpfc/kernel/execution/execution_space.hpp>
#include <openpfc/kernel/execution/layout.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <optional>
#include <stdexcept>
#include <string>

namespace pfc {

namespace detail {

template <std::size_t Rank>
std::size_t product_extents(const std::array<std::size_t, Rank> &extents) {
  std::size_t p = 1;
  for (std::size_t r = 0; r < Rank; ++r) p *= extents[r];
  return p;
}

template <typename... Args>
std::array<std::size_t, sizeof...(Args)> indices_to_array(Args... args) {
  return {{static_cast<std::size_t>(args)...}};
}

// Execution space associated with a memory space (for Kokkos compatibility)
// CudaSpace/HipSpace mappings in runtime/cuda/view_cuda.hpp and
// runtime/hip/view_hip.hpp
template <typename MemorySpace> struct memory_space_execution_space;
template <> struct memory_space_execution_space<HostSpace> { using type = Serial; };
template <typename MS>
using memory_space_execution_space_t =
    typename memory_space_execution_space<MS>::type;

} // namespace detail

/**
 * @brief Kokkos-compatible multi-dimensional View
 *
 * @tparam T Value type
 * @tparam Rank Number of dimensions (runtime extents only)
 * @tparam Layout LayoutRight or LayoutLeft
 * @tparam MemorySpace HostSpace, CudaSpace, or HipSpace
 */
template <typename T, std::size_t Rank, typename Layout = LayoutRight,
          typename MemorySpace = DefaultMemorySpace>
struct View {
  using value_type = T;
  using pointer_type = T *;
  using memory_space = MemorySpace;
  using array_layout = Layout;
  using execution_space = detail::memory_space_execution_space_t<MemorySpace>;
  using host_mirror_space = HostSpace;

  using backend_t = memory_space_to_backend_t<MemorySpace>;
  using buffer_t = core::DataBuffer<backend_t, T>;

private:
  std::optional<buffer_t> m_buffer;
  T *m_ptr = nullptr;
  std::array<std::size_t, Rank> m_extents{};
  std::array<std::size_t, Rank> m_strides{};

  static std::array<std::size_t, Rank>
  compute_strides(const std::array<std::size_t, Rank> &extents) {
    return strides_from_extents<Layout>(extents);
  }

  std::size_t linear_offset(const std::array<std::size_t, Rank> &indices) const {
    return linear_index_from_strides(indices, m_strides);
  }

public:
  /** @brief Default constructor: empty view, no allocation */
  View() = default;

  /**
   * @brief Allocating constructor
   * @param name Label for debugging/profiling (Kokkos-compatible)
   * @param extents Per-dimension extents (number of args must equal Rank)
   */
  template <typename... IntType>
  View(const std::string &name, IntType... extents)
      : m_extents{{static_cast<std::size_t>(extents)...}},
        m_strides(compute_strides(m_extents)) {
    (void)name;
    std::size_t n = detail::product_extents(m_extents);
    if (n > 0) {
      m_buffer.emplace(n);
      m_ptr = m_buffer->data();
    }
  }

  /**
   * @brief Unmanaged constructor: wrap existing pointer
   * @param ptr Pointer to existing allocation (not owned)
   * @param extents Per-dimension extents
   */
  template <typename... IntType>
  View(pointer_type ptr, IntType... extents)
      : m_ptr(ptr), m_extents{{static_cast<std::size_t>(extents)...}},
        m_strides(compute_strides(m_extents)) {}

  View(const View &) = delete;
  View &operator=(const View &) = delete;

  View(View &&other) noexcept
      : m_buffer(std::move(other.m_buffer)), m_ptr(other.m_ptr),
        m_extents(other.m_extents), m_strides(other.m_strides) {
    other.m_ptr = nullptr;
    other.m_extents = {};
    other.m_strides = {};
  }

  View &operator=(View &&other) noexcept {
    if (this != &other) {
      m_buffer = std::move(other.m_buffer);
      m_ptr = other.m_ptr;
      m_extents = other.m_extents;
      m_strides = other.m_strides;
      other.m_ptr = nullptr;
      other.m_extents = {};
      other.m_strides = {};
    }
    return *this;
  }

  static constexpr std::size_t rank() { return Rank; }
  static constexpr std::size_t rank_dynamic() { return Rank; }

  constexpr std::size_t extent(std::size_t dim) const {
    if (dim >= Rank) throw std::out_of_range("View::extent(dim)");
    return m_extents[dim];
  }

  template <typename iType> constexpr int extent_int(const iType &dim) const {
    return static_cast<int>(extent(static_cast<std::size_t>(dim)));
  }

  constexpr std::size_t stride(std::size_t dim) const {
    if (dim >= Rank) throw std::out_of_range("View::stride(dim)");
    return m_strides[dim];
  }

  constexpr std::size_t span() const { return detail::product_extents(m_extents); }

  constexpr std::size_t size() const { return detail::product_extents(m_extents); }

  constexpr pointer_type data() const { return m_ptr; }

  bool span_is_contiguous() const { return true; }

  // operator() with Rank indices (1D, 2D, 3D supported via parameter pack)
  template <typename... Args> T &operator()(Args... args) {
    static_assert(sizeof...(Args) == Rank, "View::operator() requires Rank indices");
    auto idx = detail::indices_to_array(args...);
    return m_ptr[linear_offset(idx)];
  }

  template <typename... Args> const T &operator()(Args... args) const {
    static_assert(sizeof...(Args) == Rank, "View::operator() requires Rank indices");
    auto idx = detail::indices_to_array(args...);
    return m_ptr[linear_offset(idx)];
  }

  /** @brief Kokkos-compatible access(i0, i1, ...) with defaulted trailing zeros */
  template <typename iType>
  T &access(iType i0 = 0, iType i1 = 0, iType i2 = 0, iType i3 = 0, iType i4 = 0,
            iType i5 = 0, iType i6 = 0, iType i7 = 0) {
    std::array<std::size_t, Rank> idx{};
    std::array<iType, 8> arr{{i0, i1, i2, i3, i4, i5, i6, i7}};
    for (std::size_t r = 0; r < Rank && r < 8; ++r) idx[r] = arr[r];
    return m_ptr[linear_offset(idx)];
  }

  template <typename iType>
  const T &access(iType i0 = 0, iType i1 = 0, iType i2 = 0, iType i3 = 0,
                  iType i4 = 0, iType i5 = 0, iType i6 = 0, iType i7 = 0) const {
    std::array<std::size_t, Rank> idx{};
    std::array<iType, 8> arr{{i0, i1, i2, i3, i4, i5, i6, i7}};
    for (std::size_t r = 0; r < Rank && r < 8; ++r) idx[r] = arr[r];
    return m_ptr[linear_offset(idx)];
  }

  /** @brief Whether this view owns its storage */
  bool is_managed() const { return m_buffer.has_value(); }

  /** @brief Underlying buffer (only valid when is_managed()) */
  buffer_t *buffer_ptr() { return m_buffer.has_value() ? &*m_buffer : nullptr; }
  const buffer_t *buffer_ptr() const {
    return m_buffer.has_value() ? &*m_buffer : nullptr;
  }
};

} // namespace pfc
