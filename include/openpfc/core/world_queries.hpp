// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file world_queries.hpp
 * @brief World query and coordinate transformation functions
 *
 * @details
 * This file contains functions for querying World properties and performing
 * coordinate transformations:
 *
 * **Query Functions:**
 * - get_size(), get_total_size() - Grid dimensions
 * - get_spacing(), get_origin() - Coordinate system parameters
 * - get_lower_bounds(), get_upper_bounds() - Physical domain bounds
 * - physical_volume() - Total physical volume
 * - is_1d(), is_2d(), is_3d(), dimensionality() - Dimension queries
 *
 * **Coordinate Transformations:**
 * - to_coords() - Grid indices → physical coordinates
 * - to_indices() - Physical coordinates → grid indices
 *
 * All functions are zero-cost inline operations with no runtime overhead.
 *
 * @see world.hpp for the core World struct definition
 * @see world_factory.hpp for World creation functions
 */

#pragma once

#include "csys.hpp"
#include "types.hpp"
#include "world.hpp"
#include <vector>

namespace pfc {
namespace world {

using pfc::csys::CoordinateSystem;
using pfc::types::Int3;
using pfc::types::Real3;

// ============================================================================
// Core Property Queries
// ============================================================================

/**
 * @brief Get the grid dimensions of the simulation domain
 *
 * Returns the number of grid points in each direction [nx, ny, nz].
 * This defines the discrete resolution of the computational domain.
 *
 * @tparam T Coordinate system tag (e.g., CartesianTag)
 * @param world World object to query
 * @return Grid dimensions as Int3 array [nx, ny, nz]
 *
 * @note This is the total global size before domain decomposition
 * @note For efficient FFT, dimensions should be powers of 2 or have small prime
 * factors
 *
 * @example
 * ```cpp
 * auto world = world::create({256, 128, 64});
 * Int3 size = get_size(world);
 * // size = {256, 128, 64}
 * std::cout << "Grid: " << size[0] << "×" << size[1] << "×" << size[2] << "\n";
 * ```
 *
 * Time complexity: O(1)
 * Space complexity: O(1)
 *
 * @see get_total_size() to get the product nx*ny*nz
 * @see get_size(world, index) to get size in specific dimension
 */
template <typename T> inline Int3 get_size(const World<T> &world) noexcept {
  return world.m_size;
}

/**
 * @brief Get the grid size in a specific dimension
 *
 * Returns the number of grid points in the specified dimension.
 * Convenience function equivalent to get_size(world)[index].
 *
 * @tparam T Coordinate system tag
 * @param world World object to query
 * @param index Dimension index (0=x, 1=y, 2=z)
 * @return Number of grid points in dimension 'index'
 *
 * @throws std::out_of_range if index is not 0, 1, or 2
 *
 * @note Index 0 corresponds to x-direction, 1 to y, 2 to z
 *
 * @example
 * ```cpp
 * auto world = world::create({100, 200, 300});
 * int nx = get_size(world, 0);  // Returns 100
 * int ny = get_size(world, 1);  // Returns 200
 * int nz = get_size(world, 2);  // Returns 300
 * ```
 *
 * Time complexity: O(1)
 *
 * @see get_size(world) to get all dimensions at once
 */
template <typename T> inline int get_size(const World<T> &world, int index) {
  return get_size(world).at(index);
}

/**
 * @brief Get the total number of grid points in the domain
 *
 * Computes the product nx × ny × nz, which is the total number of
 * discrete grid points in the computational domain.
 *
 * @tparam T Coordinate system tag
 * @param world World object to query
 * @return Total grid points (nx * ny * nz)
 *
 * @note This is the memory footprint for storing one scalar field
 * @note For parallel simulations, each MPI rank stores only a subset
 *
 * @example
 * ```cpp
 * auto world = world::create({128, 128, 128});
 * size_t total = get_total_size(world);
 * // total = 2,097,152 (128³)
 *
 * // Memory for double precision field:
 * size_t bytes = total * sizeof(double);
 * std::cout << "Field requires " << bytes / (1024*1024) << " MB\n";
 * ```
 *
 * Time complexity: O(1)
 * Space complexity: O(1)
 *
 * @see get_size() for individual dimensions
 * @see physical_volume() for physical domain size (not grid points)
 */
template <typename T> inline size_t get_total_size(const World<T> &world) noexcept {
  return get_size(world, 0) * get_size(world, 1) * get_size(world, 2);
}

/**
 * @brief Get the lower bounds of the world
 * @param w World object.
 * @return The lower bounds of the world.
 */
inline const auto &get_lower(const CartesianWorld &world) noexcept {
  return world.m_lower;
}

/**
 * @brief Get the lower bounds of the world in a specific dimension.
 * @param w World object.
 * @param i Dimension index.
 * @return The lower bound in the specified dimension.
 */
inline const auto &get_lower(const CartesianWorld &world, int index) {
  return get_lower(world).at(index);
}

/**
 * @brief Get the upper bounds of the world in a specific dimension.
 * @param w World object.
 * @return The upper bounds of the world.
 */
inline const auto &get_upper(const CartesianWorld &world) noexcept {
  return world.m_upper;
}

/**
 * @brief Get the upper bounds of the world in a specific dimension.
 * @param w World object.
 * @param i Dimension index.
 * @return The upper bound in the specified dimension.
 */
inline auto get_upper(const CartesianWorld &world, int index) {
  return get_upper(world).at(index);
}

// ============================================================================
// Coordinate System Queries
// ============================================================================

/**
 * @brief Get the coordinate system of the world.
 * @param w World object.
 * @return The coordinate system of the world.
 */
template <typename T>
inline const auto &get_coordinate_system(const World<T> &world) noexcept {
  return world.m_cs;
}

/**
 * @brief Get the grid spacing in all dimensions
 *
 * Returns the physical distance between adjacent grid points in each
 * direction. For periodic boundaries, spacing is (upper - lower) / size.
 * For non-periodic, spacing is (upper - lower) / (size - 1).
 *
 * @param world World object to query
 * @return Grid spacing [dx, dy, dz]
 *
 * @note Spacing determines the resolution of spectral derivatives
 * @note Smaller spacing = higher resolution but smaller time steps for stability
 * @note For FFT efficiency, uniform spacing is recommended
 *
 * @example
 * ```cpp
 * // Uniform spacing
 * auto world1 = world::create({100, 100, 100}, {0, 0, 0}, {0.1, 0.1, 0.1});
 * Real3 dx1 = get_spacing(world1);  // {0.1, 0.1, 0.1}
 *
 * // Non-uniform spacing (e.g., refined in one direction)
 * auto world2 = world::create({100, 100, 200}, {0, 0, 0}, {0.1, 0.1, 0.05});
 * Real3 dx2 = get_spacing(world2);  // {0.1, 0.1, 0.05}
 * ```
 *
 * Time complexity: O(1)
 * Space complexity: O(1)
 *
 * @see get_origin() for coordinate system offset
 * @see to_coords() which uses spacing for coordinate transforms
 */
inline const Real3 &get_spacing(const CartesianWorld &world) noexcept {
  return get_spacing(get_coordinate_system(world));
}

/**
 * @brief Get the grid spacing in a specific dimension
 *
 * Returns the physical distance between adjacent grid points in the
 * specified dimension.
 *
 * @param world World object to query
 * @param index Dimension index (0=x, 1=y, 2=z)
 * @return Grid spacing in dimension 'index'
 *
 * @throws std::out_of_range if index is not 0, 1, or 2
 *
 * @example
 * ```cpp
 * auto world = world::create({100, 100, 100}, {0, 0, 0}, {0.1, 0.2, 0.3});
 * double dx = get_spacing(world, 0);  // Returns 0.1
 * double dy = get_spacing(world, 1);  // Returns 0.2
 * double dz = get_spacing(world, 2);  // Returns 0.3
 * ```
 *
 * Time complexity: O(1)
 *
 * @see get_spacing(world) to get all spacings at once
 */
inline double get_spacing(const CartesianWorld &world, int index) noexcept {
  return get_spacing(get_coordinate_system(world), index);
}

/**
 * @brief Get the physical coordinates of the grid origin
 *
 * Returns the physical coordinates of the grid point at index (0, 0, 0).
 * This defines the offset of the coordinate system.
 *
 * @param world World object to query
 * @return Physical coordinates of origin [x0, y0, z0]
 *
 * @note The origin can be any value; common choices are (0, 0, 0) or domain center
 * @note Changing origin shifts all physical coordinates but doesn't affect
 * computation
 *
 * @example
 * ```cpp
 * // Origin at (0, 0, 0) - most common
 * auto world1 = world::create({100, 100, 100}, {0, 0, 0}, {0.1, 0.1, 0.1});
 * Real3 origin1 = get_origin(world1);  // {0, 0, 0}
 *
 * // Centered domain: origin at negative coordinates
 * auto world2 = world::create({100, 100, 100}, {-5, -5, -5}, {0.1, 0.1, 0.1});
 * Real3 origin2 = get_origin(world2);  // {-5, -5, -5}
 * // Domain spans [-5, 4.9] in each direction
 * ```
 *
 * Time complexity: O(1)
 * Space complexity: O(1)
 *
 * @see get_lower_bounds() for equivalent function
 * @see to_coords() which uses origin for coordinate transforms
 */
inline const Real3 &get_origin(const CartesianWorld &world) noexcept {
  return get_offset(get_coordinate_system(world));
}

/**
 * @brief Get the origin coordinate in a specific dimension
 *
 * Returns the physical coordinate of the grid origin in the specified dimension.
 *
 * @param world World object to query
 * @param index Dimension index (0=x, 1=y, 2=z)
 * @return Origin coordinate in dimension 'index'
 *
 * @throws std::out_of_range if index is not 0, 1, or 2
 *
 * @example
 * ```cpp
 * auto world = world::create({100, 100, 100}, {-1, 0, 5}, {0.1, 0.1, 0.1});
 * double x0 = get_origin(world, 0);  // Returns -1.0
 * double y0 = get_origin(world, 1);  // Returns 0.0
 * double z0 = get_origin(world, 2);  // Returns 5.0
 * ```
 *
 * Time complexity: O(1)
 *
 * @see get_origin(world) to get all origin coordinates
 */
inline double get_origin(const CartesianWorld &world, int index) noexcept {
  return get_offset(get_coordinate_system(world), index);
}

// ============================================================================
// Coordinate Transformations
// ============================================================================

/**
 * @brief Convert grid indices to physical coordinates
 *
 * Transforms discrete grid indices (i, j, k) to continuous physical
 * coordinates (x, y, z) based on the world's coordinate system.
 *
 * For Cartesian coordinates: x = origin[d] + i * spacing[d]
 *
 * @tparam T Coordinate system tag
 * @param world World object defining the coordinate system
 * @param indices Grid indices [i, j, k] where 0 ≤ i < nx, etc.
 * @return Physical coordinates [x, y, z]
 *
 * @note This is a zero-cost inline function (no overhead)
 * @note Indices are not bounds-checked for performance; ensure valid indices
 * @note For periodic domains, periodicity is handled by the coordinate system
 *
 * @warning Passing out-of-bounds indices results in undefined physical coordinates
 *
 * @example
 * ```cpp
 * // Create 100³ grid with 0.1 spacing starting at origin
 * auto world = world::create({100, 100, 100}, {0, 0, 0}, {0.1, 0.1, 0.1});
 *
 * // Grid point at center
 * Real3 center = to_coords(world, {50, 50, 50});
 * // center = {5.0, 5.0, 5.0}
 *
 * // Corner points
 * Real3 origin = to_coords(world, {0, 0, 0});     // {0.0, 0.0, 0.0}
 * Real3 far = to_coords(world, {99, 99, 99});     // {9.9, 9.9, 9.9}
 * ```
 *
 * Time complexity: O(1)
 * Space complexity: O(1)
 *
 * @see to_indices() for inverse transformation (physical → grid)
 * @see get_spacing() to understand coordinate scaling
 * @see get_origin() for coordinate system offset
 */
template <typename T>
inline auto to_coords(const World<T> &world, const Int3 &indices) noexcept {
  return to_coords(get_coordinate_system(world), indices);
}

/**
 * @brief Convert physical coordinates to grid indices
 *
 * Transforms continuous physical coordinates (x, y, z) to discrete grid
 * indices (i, j, k) based on the world's coordinate system.
 *
 * For Cartesian coordinates: i = round((x - origin[d]) / spacing[d])
 *
 * @tparam T Coordinate system tag
 * @param world World object defining the coordinate system
 * @param coords Physical coordinates [x, y, z]
 * @return Grid indices [i, j, k] (nearest grid point)
 *
 * @note Uses nearest-neighbor rounding (no interpolation)
 * @note For periodic domains, coordinates outside bounds wrap around
 * @note Returned indices may be outside [0, size) for non-periodic boundaries
 *
 * @warning For non-periodic domains, caller must check bounds validity
 *
 * @example
 * ```cpp
 * auto world = world::create({100, 100, 100}, {0, 0, 0}, {0.1, 0.1, 0.1});
 *
 * // Find grid point nearest to physical location
 * Real3 pos = {2.53, 4.78, 7.21};
 * Int3 idx = to_indices(world, pos);
 * // idx = {25, 48, 72} (nearest grid points)
 *
 * // Inverse transformation
 * Real3 exact = to_coords(world, idx);
 * // exact = {2.5, 4.8, 7.2} (snapped to grid)
 * ```
 *
 * @example
 * ```cpp
 * // Common usage: sample field at arbitrary position
 * auto world = world::create({64, 64, 64});
 * Field<double> density(world);
 *
 * Real3 sample_point = {31.5, 31.5, 31.5};
 * Int3 idx = to_indices(world, sample_point);
 * double value = density[idx];
 * ```
 *
 * Time complexity: O(1)
 * Space complexity: O(1)
 *
 * @see to_coords() for inverse transformation (grid → physical)
 * @see DiscreteField::interpolate() for higher-order interpolation
 */
template <typename T>
inline auto to_indices(const World<T> &world, const Real3 &coords) noexcept {
  return to_index(get_coordinate_system(world), coords);
}

// ============================================================================
// Physical Domain Queries
// ============================================================================

/**
 * @brief Compute physical volume of domain
 *
 * Returns the total physical volume (or area in 2D, length in 1D) of the
 * simulation domain.
 *
 * @param world World instance
 * @return Physical volume V = Lx * Ly * Lz where L = spacing * size
 *
 * @note For Cartesian coordinates, this is the product of all physical dimensions
 * @note Works correctly for 1D, 2D, and 3D domains
 *
 * @code
 * auto world = world::create({100, 100, 100}, {0, 0, 0}, {0.1, 0.1, 0.1});
 * double vol = world::physical_volume(world);  // Returns 1000.0
 * @endcode
 *
 * Time complexity: O(1)
 * Space complexity: O(1)
 */
template <typename T> inline double physical_volume(const World<T> &world) noexcept {
  const auto spacing = get_spacing(world);
  const auto size = get_size(world);
  return spacing[0] * spacing[1] * spacing[2] * size[0] * size[1] * size[2];
}

/**
 * @brief Check if domain is 1D (only x-direction has > 1 point)
 *
 * A domain is considered 1D if only the first dimension has more than
 * one grid point.
 *
 * @param world World instance
 * @return true if only x-direction is active (nx > 1, ny = 1, nz = 1)
 *
 * @code
 * auto world1d = world::create({100, 1, 1});
 * bool is_1d = world::is_1d(world1d);  // Returns true
 * @endcode
 *
 * Time complexity: O(1)
 */
template <typename T> inline bool is_1d(const World<T> &world) noexcept {
  const auto size = get_size(world);
  return (size[0] > 1) && (size[1] == 1) && (size[2] == 1);
}

/**
 * @brief Check if domain is 2D (x and y have > 1 point, z has 1)
 *
 * A domain is considered 2D if the first two dimensions have more than
 * one grid point and the third has exactly one.
 *
 * @param world World instance
 * @return true if x and y directions are active (nx > 1, ny > 1, nz = 1)
 *
 * @code
 * auto world2d = world::create({64, 64, 1});
 * bool is_2d = world::is_2d(world2d);  // Returns true
 * @endcode
 *
 * Time complexity: O(1)
 */
template <typename T> inline bool is_2d(const World<T> &world) noexcept {
  const auto size = get_size(world);
  return (size[0] > 1) && (size[1] > 1) && (size[2] == 1);
}

/**
 * @brief Check if domain is 3D (all dimensions have > 1 point)
 *
 * A domain is considered 3D if all three dimensions have more than
 * one grid point.
 *
 * @param world World instance
 * @return true if all three directions are active (nx > 1, ny > 1, nz > 1)
 *
 * @code
 * auto world3d = world::create({32, 32, 32});
 * bool is_3d = world::is_3d(world3d);  // Returns true
 * @endcode
 *
 * Time complexity: O(1)
 */
template <typename T> inline bool is_3d(const World<T> &world) noexcept {
  const auto size = get_size(world);
  return (size[0] > 1) && (size[1] > 1) && (size[2] > 1);
}

/**
 * @brief Get dimensionality as integer
 *
 * Returns 1, 2, or 3 based on how many dimensions have more than one
 * grid point. Returns 0 for degenerate case where all dimensions have
 * size 1.
 *
 * @param world World instance
 * @return 1, 2, 3, or 0 (degenerate) based on active dimensions
 *
 * @code
 * auto world2d = world::create({64, 64, 1});
 * int dim = world::dimensionality(world2d);  // Returns 2
 * @endcode
 *
 * Time complexity: O(1)
 */
template <typename T> inline int dimensionality(const World<T> &world) noexcept {
  if (is_3d(world)) return 3;
  if (is_2d(world)) return 2;
  if (is_1d(world)) return 1;
  return 0; // Degenerate case (all dimensions size 1)
}

/**
 * @brief Get physical lower bounds (origin corner)
 *
 * Returns the physical coordinates of the grid point at index (0, 0, 0).
 * This is the lower corner of the domain.
 *
 * @param world World instance
 * @return Physical coordinates of (0, 0, 0) grid point
 *
 * @code
 * auto world = world::create({100, 100, 100}, {-5, -5, 0}, {0.1, 0.1, 0.1});
 * Real3 lower = world::get_lower_bounds(world);  // Returns {-5, -5, 0}
 * @endcode
 *
 * Time complexity: O(1)
 */
template <typename T> inline Real3 get_lower_bounds(const World<T> &world) noexcept {
  return to_coords(world, {0, 0, 0});
}

/**
 * @brief Get physical upper bounds (far corner)
 *
 * Returns the physical coordinates of the grid point at the maximum indices
 * (nx-1, ny-1, nz-1). This is the upper corner of the domain.
 *
 * @param world World instance
 * @return Physical coordinates of (nx-1, ny-1, nz-1) grid point
 *
 * @code
 * auto world = world::create({100, 100, 100}, {0, 0, 0}, {0.1, 0.1, 0.1});
 * Real3 upper = world::get_upper_bounds(world);  // Returns {9.9, 9.9, 9.9}
 * @endcode
 *
 * Time complexity: O(1)
 */
template <typename T> inline Real3 get_upper_bounds(const World<T> &world) noexcept {
  const auto size = get_size(world);
  return to_coords(world, {size[0] - 1, size[1] - 1, size[2] - 1});
}

} // namespace world
} // namespace pfc

namespace pfc {
namespace world {

/**
 * @brief Generate per-axis coordinate arrays for the world grid
 *
 * Returns three arrays containing the physical coordinates along each axis
 * (x, y, z). Each array has length equal to the grid size in that dimension.
 *
 * @tparam T Coordinate system tag
 * @param world World object to query
 * @return std::array of 3 vectors: {x_coords, y_coords, z_coords}
 *
 * @example
 * ```cpp
 * auto w = world::create({16,16,16});
 * auto coords = world::coordinates(w);
 * // coords[0][i] = x(i), coords[1][j] = y(j), coords[2][k] = z(k)
 * ```
 */
template <typename T>
inline std::array<std::vector<double>, 3>
coordinates(const World<T> &world) noexcept {
  std::array<std::vector<double>, 3> result;
  const auto size = get_size(world);
  const auto origin = get_origin(world);
  const auto spacing = get_spacing(world);
  for (int d = 0; d < 3; ++d) {
    result[d].resize(size[d]);
    for (int i = 0; i < size[d]; ++i) {
      result[d][i] = origin[d] + i * spacing[d];
    }
  }
  return result;
}

} // namespace world
} // namespace pfc
