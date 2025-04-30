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
 * @note The World class is designed to be used in conjunction with other
 * components of the framework, such as Decomposition and ArrayND, to provide
 * a complete simulation environment.
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

/// Coordinate system tag.
enum class CoordinateSystemTag {
  Line,        ///< 1D Cartesian
  Plane,       ///< 2D Cartesian
  Cartesian,   ///< 3D Cartesian
  Polar,       ///< 2D Polar
  Cylindrical, ///< 3D Cylindrical
  Spherical    ///< 3D Spherical
};

/// Strong typedefs for constructor clarity
struct Size3 {
  std::array<int, 3> value;
  explicit Size3(const std::array<int, 3> &v) : value(v) {
    for (int dim : value) {
      if (dim <= 0) {
        throw std::invalid_argument("Size values must be positive.");
      }
    }
  }
};

struct LowerBounds3 {
  std::array<double, 3> value;
  explicit LowerBounds3(const std::array<double, 3> &v) : value(v) {}
};

struct UpperBounds3 {
  std::array<double, 3> value;
  explicit UpperBounds3(const std::array<double, 3> &v) : value(v) {}
};

struct Spacing3 {
  std::array<double, 3> value;
  explicit Spacing3(const std::array<double, 3> &v) : value(v) {
    for (double dim : value) {
      if (dim <= 0.0) {
        throw std::invalid_argument("Spacing values must be positive.");
      }
    }
  }
};

struct Periodic3 {
  std::array<bool, 3> value;
  explicit Periodic3(const std::array<bool, 3> &v) : value(v) {}
};

/**
 * @brief Represents the global simulation domain (the "world").
 *
 * The World class defines the *size*, *origin*, and *grid spacing* of the
 * global simulation domain. It provides an abstraction for physical (x, y, z)
 * space and the corresponding (i, j, k) grid indices.
 *
 *
 * ## Responsibilities
 *
 * - Encapsulates the *geometric* description of the global domain.
 * - Provides basic *coordinate transformations* between grid indices and
 *   physical coordinates.
 * - Ensures that domain sizes and spacings are valid (positive, consistent).
 *
 *
 * ## Design Justification
 *
 * - Decouples physical space information from other components (fields,
 *   solvers, decomposition).
 * - Allows flexible handling of different domain sizes, resolutions, and
 *   coordinate systems.
 * - Keeps simulation setup and post-processing simpler by clearly defining the
 *   world geometry.
 *
 *
 * ## Relations to Other Components
 *
 * - Used by Decomposition to partition the domain across processes.
 * - Used by ArrayND to associate data with physical coordinates.
 *
 *
 * ## Example usage
 *
 * @code
 * #include "openpfc/core/world.hpp"
 * #include <iostream>
 *
 * int main() {
 *     pfc::World::Int3 dimensions = {100, 100, 100};
 *     pfc::World::Real3 origin = {0.0, 0.0, 0.0};
 *     pfc::World::Real3 spacing = {1.0, 1.0, 1.0};
 *     pfc::World world(dimensions, origin, spacing);
 *
 *     auto coords = world.physical_coordinates({10, 20, 30});
 *     auto indices = world.grid_indices({10.0, 20.0, 30.0});
 *
 *     std::cout << world << std::endl;
 * }
 * @endcode
 */
struct World final {
  const Int3 m_size;      ///< Dimensions of the world: {Lx, Ly, Lz}
  const Real3 m_lower;    ///< Lower coordinates: {x0, y0, z0}
  const Real3 m_upper;    ///< Upper coordinates: {x1, y1, z1}
  const Real3 m_spacing;  ///< Spacing parameters: {dx, dy, dz}
  const Bool3 m_periodic; ///< Periodicity flags: {px, py, pz}
  const CoordinateSystemTag m_coordinate_system; ///< Coordinate system type

  // constructor
  explicit World(const Int3 &dimensions, const Real3 &lower, const Real3 &upper,
                 const Real3 &spacing, const Bool3 &periodic,
                 CoordinateSystemTag coordinate_system);

  // comparison operators
  bool operator==(const World &other) const noexcept;
  bool operator!=(const World &other) const noexcept;

  // stream output operator
  friend std::ostream &operator<<(std::ostream &os, const World &w) noexcept;
};

/**
 * @brief Create a World object with the specified dimensions, origin, and
 * spacing.
 */
World create_world(const Int3 &dimensions, const Real3 &origin,
                   const Real3 &spacing);

/**
 * @brief Create a World object with the specified dimensions and default
 * origin and spacing.
 *
 * Default origin is {0.0, 0.0, 0.0}, and default spacing is {1.0, 1.0, 1.0}.
 */
World create_world(const Int3 &dimensions);

Int3 get_size(const World &w) noexcept;
size_t get_size(const World &w, int i) noexcept;

Real3 get_origin(const World &w) noexcept;
double get_origin(const World &w, int i) noexcept;

Real3 get_lower(const World &w) noexcept;
double get_lower(const World &w, int i) noexcept;

Real3 get_upper(const World &w) noexcept;
double get_upper(const World &w, int i) noexcept;

Real3 get_spacing(const World &w) noexcept;
double get_spacing(const World &w, int i) noexcept;

/**
 * @brief Get the number of grid points in each dimension.
 *
 * @return The number of grid points in each dimension: {Nx, Ny, Nz}.
 */
int total_size(const World &w) noexcept;

/**
 * @brief Computes the physical coordinate corresponding to grid indices
 * {i,j,k}.
 *
 * This method calculates the physical coordinates in the simulation domain
 * based on the grid indices and spacing.
 *
 * @param indices The grid indices {i, j, k}.
 * @return The physical coordinate {x, y, z}.
 */
Real3 to_coords(const World &w, const Int3 &indices) noexcept;

/**
 * @brief Computes the grid indices corresponding to physical coordinates
 * {x,y,z}.
 *
 * This method calculates the grid indices in the simulation domain
 * based on the physical coordinates and spacing.
 *
 * @param coordinates The physical coordinates {x, y, z}.
 * @return The grid indices {i, j, k}.
 */
Int3 to_indices(const World &w, const Real3 &coordinates) noexcept;

// Free function API for coordinate system and periodicity
CoordinateSystemTag get_coordinate_system(const World &w) noexcept;

const Bool3 &get_periodicity(const World &w) noexcept;

const bool &is_periodic(const World &w, int i) noexcept;

// Free function variations for creating a World object
World create_world(const Size3 &size, const LowerBounds3 &lower,
                   const UpperBounds3 &upper);

World create_world(const Size3 &size, const UpperBounds3 &upper);

World create_world(const Size3 &size, const LowerBounds3 &lower,
                   const Spacing3 &spacing);

} // namespace pfc
