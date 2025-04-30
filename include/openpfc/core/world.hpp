// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file world.hpp
 * @brief World class definition
 * @details This file contains the definition of the World class, which
 * represents the global simulation domain in a computational physics
 * framework. The World class encapsulates the size, origin, and grid spacing
 * of the simulation domain, and provides methods for coordinate transformations
 * between grid indices and physical coordinates.
 *
 * The World class is designed to be used in conjunction with various
 * coordinate systems (Cartesian, Polar, Cylindrical, Spherical) and supports
 * periodic boundary conditions. It provides a clear and consistent interface
 * for working with simulation domains, making it easier to manage and
 * manipulate the underlying data structures.
 */

#pragma once

#include <array>
#include <ostream>
#include <stdexcept>

namespace pfc {

/// Type aliases for clarity
using Int3 = std::array<int, 3>;
using Real3 = std::array<double, 3>;
using Bool3 = std::array<bool, 3>;

/**
 * @brief Coordinate system tag.
 */
enum class CoordinateSystemTag {
  Line,        ///< 1D Cartesian
  Plane,       ///< 2D Cartesian
  Cartesian,   ///< 3D Cartesian
  Polar,       ///< 2D Polar
  Cylindrical, ///< 3D Cylindrical
  Spherical    ///< 3D Spherical
};

// Strong typedefs for constructor clarity
/**
 * @brief Represents the size of the simulation domain.
 */
struct Size3 {
  std::array<int, 3> value;
  explicit Size3(const std::array<int, 3> &v);
};

/**
 * @brief Represents the lower bounds of the simulation domain.
 */
struct LowerBounds3 {
  std::array<double, 3> value;
  explicit LowerBounds3(const std::array<double, 3> &v);
};

/**
 * @brief Represents the upper bounds of the simulation domain.
 */
struct UpperBounds3 {
  std::array<double, 3> value;
  explicit UpperBounds3(const std::array<double, 3> &v);
};

/**
 * @brief Represents the spacing of the simulation grid.
 */
struct Spacing3 {
  std::array<double, 3> value;
  explicit Spacing3(const std::array<double, 3> &v);
};

/**
 * @brief Represents the periodicity of the simulation domain.
 */
struct Periodic3 {
  std::array<bool, 3> value;
  explicit Periodic3(const std::array<bool, 3> &v);
};

/**
 * @brief Represents the global simulation domain (the "world").
 *
 * The World class defines the *size*, *origin*, and *grid spacing* of the
 * global simulation domain. It provides an abstraction for physical (x, y, z)
 * space and the corresponding (i, j, k) grid indices.
 */
struct World final {
  const Int3 m_size;      ///< Dimensions of the world: {Lx, Ly, Lz}
  const Real3 m_lower;    ///< Lower coordinates: {x0, y0, z0}
  const Real3 m_upper;    ///< Upper coordinates: {x1, y1, z1}
  const Real3 m_spacing;  ///< Spacing parameters: {dx, dy, dz}
  const Bool3 m_periodic; ///< Periodicity flags: {px, py, pz}
  const CoordinateSystemTag m_coordinate_system; ///< Coordinate system type

  /**
   * @brief Constructs a World object.
   * @param dimensions Dimensions of the world.
   * @param lower Lower bounds of the world.
   * @param upper Upper bounds of the world.
   * @param spacing Spacing of the grid.
   * @param periodic Periodicity flags.
   * @param coordinate_system Coordinate system type.
   */
  explicit World(const Int3 &dimensions, const Real3 &lower, const Real3 &upper,
                 const Real3 &spacing, const Bool3 &periodic,
                 CoordinateSystemTag coordinate_system);

  /**
   * @brief Equality operator.
   * @param other Another World object.
   * @return True if equal, false otherwise.
   */
  bool operator==(const World &other) const noexcept;

  /**
   * @brief Inequality operator.
   * @param other Another World object.
   * @return True if not equal, false otherwise.
   */
  bool operator!=(const World &other) const noexcept;

  /**
   * @brief Stream output operator.
   * @param os Output stream.
   * @param w World object.
   * @return Reference to the output stream.
   */
  friend std::ostream &operator<<(std::ostream &os, const World &w) noexcept;
};

// Free function API for creating World objects

/**
 * @brief Create a World object with the specified dimensions, origin, and spacing.
 * @param dimensions Dimensions of the world.
 * @param origin Origin of the world.
 * @param spacing Spacing of the grid.
 * @return A World object.
 */
World create_world(const Int3 &dimensions, const Real3 &origin,
                   const Real3 &spacing);

/**
 * @brief Create a World object with the specified dimensions and default origin and
 * spacing.
 * @param dimensions Dimensions of the world.
 * @return A World object.
 */
World create_world(const Int3 &dimensions);

/**
 * @brief Create a World object with the specified size, lower bounds, upper bounds,
 * spacing, and periodicity.
 * @param size Size of the world.
 * @param lower Lower bounds of the world.
 * @param upper Upper bounds of the world.
 * @param spacing Spacing of the grid.
 * @param periodic Periodicity flags.
 * @param cs Coordinate system type.
 * @return A World object.
 */
World create_world(const Size3 &size, const LowerBounds3 &lower,
                   const UpperBounds3 &upper, const Spacing3 &spacing,
                   const Periodic3 &periodic, const CoordinateSystemTag &cs);

/**
 * @brief Create a World object with the specified size, lower bounds, upper bounds,
 * periodicity, and coordinate system.
 * @param size Size of the world.
 * @param lower Lower bounds of the world.
 * @param upper Upper bounds of the world.
 * @param periodic Periodicity flags.
 * @param cs Coordinate system type.
 * @return A World object.
 */
World create_world(const Size3 &size, const LowerBounds3 &lower,
                   const UpperBounds3 &upper, const Periodic3 &periodic,
                   const CoordinateSystemTag &cs);

/**
 * @brief Create a World object with the specified size, lower bounds, spacing,
 * periodicity, and coordinate system.
 * @param size Size of the world.
 * @param lower Lower bounds of the world.
 * @param spacing Spacing of the grid.
 * @param periodic Periodicity flags.
 * @param cs Coordinate system type.
 * @return A World object.
 */
World create_world(const Size3 &size, const LowerBounds3 &lower,
                   const Spacing3 &spacing, const Periodic3 &periodic,
                   const CoordinateSystemTag &cs);

/**
 * @brief Create a World object with the specified size and upper bounds.
 * @param size Size of the world.
 * @param upper Upper bounds of the world.
 * @return A World object.
 */
World create_world(const Size3 &size, const UpperBounds3 &upper);

// Free function API for querying World properties

/**
 * @brief Get the size of the world.
 * @param w World object.
 * @return The size of the world.
 */
Int3 get_size(const World &w) noexcept;

/**
 * @brief Get the size of the world in a specific dimension.
 * @param w World object.
 * @param i Dimension index.
 * @return The size in the specified dimension.
 */
size_t get_size(const World &w, int i) noexcept;

/**
 * @brief Get the origin of the world.
 * @param w World object.
 * @return The origin of the world.
 */
Real3 get_origin(const World &w) noexcept;

/**
 * @brief Get the origin of the world in a specific dimension.
 * @param w World object.
 * @param i Dimension index.
 * @return The origin in the specified dimension.
 */
double get_origin(const World &w, int i) noexcept;

/**
 * @brief Get the lower bounds of the world.
 * @param w World object.
 * @return The lower bounds of the world.
 */
Real3 get_lower(const World &w) noexcept;

/**
 * @brief Get the lower bounds of the world in a specific dimension.
 * @param w World object.
 * @param i Dimension index.
 * @return The lower bound in the specified dimension.
 */
double get_lower(const World &w, int i) noexcept;

/**
 * @brief Get the upper bounds of the world.
 * @param w World object.
 * @return The upper bounds of the world.
 */
Real3 get_upper(const World &w) noexcept;

/**
 * @brief Get the upper bounds of the world in a specific dimension.
 * @param w World object.
 * @param i Dimension index.
 * @return The upper bound in the specified dimension.
 */
double get_upper(const World &w, int i) noexcept;

/**
 * @brief Get the spacing of the world.
 * @param w World object.
 * @return The spacing of the world.
 */
Real3 get_spacing(const World &w) noexcept;

/**
 * @brief Get the spacing of the world in a specific dimension.
 * @param w World object.
 * @param i Dimension index.
 * @return The spacing in the specified dimension.
 */
double get_spacing(const World &w, int i) noexcept;

/**
 * @brief Get the total number of grid points in the world.
 * @param w World object.
 * @return The total number of grid points.
 */
int total_size(const World &w) noexcept;

/**
 * @brief Compute the physical coordinates corresponding to grid indices.
 * @param w World object.
 * @param indices Grid indices.
 * @return The physical coordinates.
 */
Real3 to_coords(const World &w, const Int3 &indices) noexcept;

/**
 * @brief Compute the grid indices corresponding to physical coordinates.
 * @param w World object.
 * @param coordinates Physical coordinates.
 * @return The grid indices.
 */
Int3 to_indices(const World &w, const Real3 &coordinates) noexcept;

// Free function API for coordinate system and periodicity
/**
 * @brief Get the coordinate system of the world.
 * @param w World object.
 * @return The coordinate system tag.
 */
CoordinateSystemTag get_coordinate_system(const World &w) noexcept;

/**
 * @brief Get the periodicity of the world.
 * @param w World object.
 * @return The periodicity flags.
 */
const Bool3 &get_periodicity(const World &w) noexcept;

/**
 * @brief Check if the world is periodic in a specific dimension.
 * @param w World object.
 * @param i Dimension index.
 * @return True if periodic, false otherwise.
 */
bool is_periodic(const World &w, int i) noexcept;

} // namespace pfc
