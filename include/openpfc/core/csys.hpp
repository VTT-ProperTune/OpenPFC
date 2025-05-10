// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include "types.hpp"

namespace pfc {
namespace csys {

using pfc::types::Bool3;
using pfc::types::Int3;
using pfc::types::Real3;

/**
 * @brief Primary template for defining coordinate systems by tag.
 *
 * @details
 * The `CoordinateSystem<Tag>` template provides a mechanism to define coordinate
 * systems in a modular and extensible way, where each `Tag` corresponds to a
 * specific geometry (e.g., Cartesian, Polar, Cylindrical).
 *
 * Specializations of this template should define:
 * - The internal parameters of the coordinate system (e.g., offset, spacing)
 * - Methods to map between index space and physical space:
 *   - `to_physical(const Int3&) -> Real3`
 *   - `to_index(const Real3&) -> Int3`
 *
 * This design decouples geometry from logic and enables user-defined coordinate
 * systems to integrate cleanly with the simulation framework. It also avoids
 * inheritance or runtime polymorphism, favoring compile-time specialization and
 * inlining for performance-critical transformations.
 *
 * Example usage:
 * @code
 * struct CartesianTag {};
 *
 * template <>
 * struct CoordinateSystem<CartesianTag> {
 *   Real3 offset;
 *   Real3 spacing;
 *
 *   Real3 to_physical(const Int3& idx) const noexcept;
 *   Int3 to_index(const Real3& pos) const noexcept;
 * };
 * @endcode
 *
 * Coordinate systems are used by the `World<Tag>` class and related infrastructure
 * to define how grid indices map to real-world physical coordinates.
 *
 * @tparam Tag A user-defined type that uniquely identifies the coordinate system.
 */
template <typename Tag> struct CoordinateSystem;

/**
 * @brief Trait class for providing default parameters for coordinate systems.
 *
 * @details
 * This traits template defines default values (e.g., offset, spacing, periodicity)
 * for coordinate systems identified by their tag type `CoordTag`.
 *
 * The primary template is left undefined, and users are expected to specialize
 * this trait for their own coordinate system tags to provide meaningful defaults.
 *
 * This mechanism allows external extension of coordinate system behavior
 * without modifying the core library, enabling user-defined systems to
 * seamlessly integrate with generic `World` construction and other infrastructure.
 *
 * Example specialization:
 * @code
 * struct MyCoordTag {};
 *
 * template <>
 * struct CoordinateSystemDefaults<MyCoordTag> {
 *   static constexpr Real3 offset = {0.0, 0.0, 0.0};
 *   static constexpr Real3 spacing = {1.0, 1.0, 1.0};
 *   static constexpr Bool3 periodicity = {true, false, false};
 * };
 * @endcode
 */
template <typename CoordTag>
struct CoordinateSystemDefaults; // intentionally left undefined

// Coordinate system tags.
// struct LineTag {};
// struct PlaneTag {};
// struct PolarTag {};
// struct CylindricalTag {};
// struct SphericalTag {};
// struct Polar2DTag {};
// struct Toroidal2DTag {};
// struct LogPolar3DTag {};

/**
 * @brief Tag type for the 3D Cartesian coordinate system.
 *
 * @details
 * This tag represents a standard right-handed Cartesian coordinate system in 3D,
 * which is the default and most commonly used geometry in OpenPFC simulations.
 *
 * The associated coordinate system maps discrete grid indices (i, j, k) to
 * continuous physical space using a uniform, axis-aligned grid defined by:
 * - `offset`: the physical position corresponding to index (0, 0, 0)
 * - `spacing`: the distance between adjacent grid points along each axis
 *
 * This coordinate system assumes:
 * - The grid is regular (uniform spacing)
 * - Axes are orthogonal and aligned with the simulation dimensions
 *
 * It provides a simple and efficient mapping suitable for a wide range of
 * physical simulations where a Euclidean space is appropriate.
 */
struct CartesianTag {};

/**
 * @brief Default parameters for the 3D Cartesian coordinate system.
 *
 * @details
 * This specialization of `CoordinateSystemDefaults` provides the default values
 * used when constructing a 3D Cartesian coordinate system (`CartesianTag`)
 * without explicitly specifying offset, spacing, or periodicity.
 *
 * These defaults are consistent with standard simulation conventions:
 * - `offset = {0.0, 0.0, 0.0}` places the (0, 0, 0) index at the physical origin.
 * - `spacing = {1.0, 1.0, 1.0}` defines unit grid spacing in all dimensions.
 * - `periodicity = {true, true, true}` models a fully periodic domain.
 *
 * These values are used by factory functions and world constructors when
 * coordinate system parameters are not explicitly provided by the user.
 */
template <> struct CoordinateSystemDefaults<CartesianTag> {
  static constexpr Real3 offset = {0.0, 0.0, 0.0};
  static constexpr Real3 spacing = {1.0, 1.0, 1.0};
  static constexpr Bool3 periodic = {true, true, true};
  std::size_t dimensions = 3; ///< Number of dimensions in the coordinate system
};

/**
 * @brief Specialization of the coordinate system for 3D Cartesian space.
 *
 * @details
 * This structure defines a uniform, axis-aligned Cartesian coordinate system
 * in three dimensions. It maps discrete grid indices (i, j, k) to continuous
 * physical coordinates using a regular grid geometry.
 *
 * The coordinate system is defined by three key parameters:
 * - `m_offset`: the physical position corresponding to the grid index (0, 0, 0)
 * - `m_spacing`: the uniform distance between adjacent grid points in each dimension
 * - `m_periodic`: flags indicating periodicity along each axis
 *
 * This specialization is used internally by the `World<CartesianTag>` type
 * and can be constructed directly or via factory methods. It is designed to
 * support fast, inlined coordinate transformations for performance-critical
 * simulation kernels.
 *
 * @constructor
 * Constructs a Cartesian coordinate system with the specified offset, spacing,
 * and periodicity. Throws `std::invalid_argument` if any spacing component
 * is non-positive.
 *
 * @param offset     Physical position of the index (0, 0, 0)
 * @param spacing    Grid spacing in each dimension (must be > 0)
 * @param periodic   Periodicity flags for {x, y, z}
 */
template <> struct CoordinateSystem<CartesianTag> {
  const Real3 m_offset;   ///< Physical coordinate of grid index (0, 0, 0)
  const Real3 m_spacing;  ///< Physical spacing between grid points
  const Bool3 m_periodic; ///< Periodicity flags for each dimension

  /// Constructs a 3D Cartesian coordinate system
  CoordinateSystem(
      const Real3 &offset = CoordinateSystemDefaults<CartesianTag>::offset,
      const Real3 &spacing = CoordinateSystemDefaults<CartesianTag>::spacing,
      const Bool3 &periodic = CoordinateSystemDefaults<CartesianTag>::periodic)
      : m_offset(offset), m_spacing(spacing), m_periodic(periodic) {
    for (std::size_t i = 0; i < 3; ++i) {
      if (spacing[i] <= 0.0) {
        throw std::invalid_argument("Spacing must be positive.");
      }
    };
  };
};

using CartesianCS = CoordinateSystem<CartesianTag>;

/**
 * @brief Get the offset of the coordinate system.
 * @param cs Coordinate system object.
 * @return The offset of the coordinate system.
 */
inline const Real3 &get_offset(const CartesianCS &cs) noexcept {
  return cs.m_offset;
};

/**
 * @brief Get the offset of the coordinate system in a specific dimension.
 * @param cs Coordinate system object.
 * @param i Dimension index.
 * @return The offset in the specified dimension.
 * @throws std::out_of_range if i is not in [0, 2].
 */
inline double get_offset(const CartesianCS &cs, int i) { return cs.m_offset.at(i); }

/**
 * @brief Get the spacing of the coordinate system.
 * @param cs Coordinate system object.
 * @return The spacing of the coordinate system.
 */
inline const Real3 &get_spacing(const CartesianCS &cs) noexcept {
  return cs.m_spacing;
};

/**
 * @brief Get the spacing of the coordinate system in a specific dimension.
 * @param cs Coordinate system object.
 * @param i Dimension index.
 * @return The spacing in the specified dimension.
 * @throws std::out_of_range if i is not in [0, 2].
 */
inline double get_spacing(const CartesianCS &cs, int i) {
  return cs.m_spacing.at(i);
}

/**
 * @brief Get the periodicity of the coordinate system.
 * @param cs Coordinate system object.
 * @return The periodicity flags.
 */
inline const Bool3 &get_periodic(const CartesianCS &cs) noexcept {
  return cs.m_periodic;
};

/**
 * @brief Check if the coordinate system is periodic in a specific dimension.
 * @param cs Coordinate system object.
 * @param i Dimension index.
 * @return True if periodic, false otherwise.
 * @throws std::out_of_range if i is not in [0, 2].
 */
inline bool is_periodic(const CartesianCS &cs, int i) { return cs.m_periodic.at(i); }

/**
 * @brief Convert grid indices to physical coordinates.
 * @param cs Coordinate system object.
 * @param idx Grid indices.
 * @return The physical coordinates.
 */
inline const Real3 to_coords(const CartesianCS &cs, const Int3 &idx) noexcept {
  Real3 xyz;
  const auto &offset = get_offset(cs);
  const auto &spacing = get_spacing(cs);
  for (int i = 0; i < 3; ++i) {
    xyz[i] = offset[i] + idx[i] * spacing[i];
  }
  return xyz;
}

/**
 * @brief Convert physical coordinates to grid indices.
 * @param cs Coordinate system object.
 * @param xyz Physical coordinates.
 * @return The grid indices.
 */
inline const Int3 to_index(const CartesianCS &cs, const Real3 &xyz) noexcept {
  Int3 idx;
  const auto &offset = get_offset(cs);
  const auto &spacing = get_spacing(cs);
  for (int i = 0; i < 3; ++i) {
    idx[i] = static_cast<int>((xyz[i] - offset[i]) / spacing[i]);
  }
  return idx;
}

} // namespace csys
} // namespace pfc
