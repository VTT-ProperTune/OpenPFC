// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file fields.hpp
 * @brief Defines the Field<T> structure and related free functions for data storage
 * in OpenPFC.
 *
 * Fields in OpenPFC represent structured data over a local simulation domain.
 * A Field is parameterized by the value type `T`, and its layout is defined by
 * an immutable `MultiIndex<3>` and `CoordinateSystem`, both derived from the local
 * World.
 *
 * ## Design Principles
 *
 * - Fields are implemented as open `struct Field<T>` by default.
 * - Members such as `m_data`, `m_index`, and `m_coordsys` are publicly accessible
 * but prefixed with `m_` to indicate internal use. Users may access these directly
 * but are encouraged to prefer free functions.
 * - Field size and layout are fixed upon construction and cannot be resized.
 * - Element values are mutable, but the domain shape and coordinate mapping are
 * immutable.
 * - All functionality (e.g. data access, coordinate transforms, fill, copy) is
 * provided via free functions in the `pfc::field` namespace.
 *
 * ## Usage Example
 *
 * ```cpp
 * auto world = get_subworld(decomposition, rank_id);
 * auto field = field::create<double>(world);
 *
 * field::fill(field, [](double x, double y, double z) {
 *     return std::exp(-(x*x + y*y + z*z));
 * });
 *
 * double value = field::at(field, {i, j, k});
 * Real3 pos = to_physical(get_coordsys(field), {i, j, k});
 * ```
 *
 * Fields support scalar types like `double`, but also custom vector or tensor
 * types, e.g., `Vec3f32`, as long as they are trivially copyable and can be stored
 * in `std::vector<T>`.
 *
 * This component is designed to work seamlessly with differential operators,
 * FFTs, and models without ever exposing internal logic unnecessarily.
 */

#include "openpfc/core/csys.hpp"
#include "openpfc/core/types.hpp"
#include "openpfc/core/world.hpp"
#include <array>
#include <cassert>
#include <functional>
#include <vector>

namespace pfc {
namespace field {

using World = pfc::world::World<pfc::csys::CartesianTag>;
using pfc::world::get_total_size;

template <typename T> struct Field {
  std::vector<T> m_data;
  const World &m_world;

  Field(const World &world) : m_data(get_total_size(world)), m_world(world) {
    assert(m_data.size() > 0);
  }

  Field(const Field &) = delete;
  Field &operator=(const Field &) = delete;

  Field(Field &&) = default;
  Field &operator=(Field &&) = delete;
};

template <typename T> Field<T> create(const World &world) { return Field<T>(world); }

template <typename T> inline const auto &get_data(const Field<T> &field) {
  return field.m_data;
}

template <typename T> inline auto &get_data(Field<T> &field) { return field.m_data; }

template <typename T> inline const auto &get_world(const Field<T> &field) {
  return field.m_world;
}

} // namespace field
} // namespace pfc
