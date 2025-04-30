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
class World {
public:
  /// Type aliases for clarity
  using Int3 = std::array<int, 3>;
  using Real3 = std::array<double, 3>;

private:
  const Int3 m_size;     ///< Dimensions of the world: {Lx, Ly, Lz}
  const Real3 m_origin;  ///< Origin coordinates: {x0, y0, z0}
  const Real3 m_spacing; ///< Spacing parameters: {dx, dy, dz}

public:
  World(const Int3 &dimensions, const Real3 &origin, const Real3 &spacing)
      : m_size(dimensions), m_origin(origin), m_spacing(spacing) {}

  // Getters for member variables
  Int3 get_size() const noexcept { return m_size; }
  Real3 get_origin() const noexcept { return m_origin; }
  Real3 get_spacing() const noexcept { return m_spacing; }

  /**
   * @brief Get the number of grid points in each dimension.
   *
   * @return The number of grid points in each dimension: {Nx, Ny, Nz}.
   */
  int total_size() const noexcept;

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
  Real3 physical_coordinates(const Int3 &indices) const noexcept;

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
  Int3 grid_indices(const Real3 &coordinates) const noexcept;

  /**
   * @brief Compare this world to other world.
   *
   * @param other world
   * @return true
   * @return false
   */
  bool operator==(const World &other) const noexcept;

  /**
   * @brief Compare this world to other world.
   *
   * @param other world
   * @return true
   * @return false
   */
  bool operator!=(const World &other) const noexcept;

  /**
   * @brief Output stream operator for World objects.
   *
   * Allows printing the state of a World object to an output stream.
   *
   * @param os The output stream to write to.
   * @param w The World object to be printed.
   * @return The updated output stream.
   */
  friend std::ostream &operator<<(std::ostream &os, const World &w) noexcept;
};

/**
 * @brief Create a World object with the specified dimensions, origin, and
 * spacing.
 */
World create_world(const World::Int3 &dimensions, const World::Real3 &origin,
                   const World::Real3 &spacing);

/**
 * @brief Create a World object with the specified dimensions and default
 * origin and spacing.
 *
 * Default origin is {0.0, 0.0, 0.0}, and default spacing is {1.0, 1.0, 1.0}.
 */
World create_world(const World::Int3 &dimensions);

World::Int3 get_size(const World &world) noexcept;
size_t get_size(const World &world, int dim) noexcept;
World::Real3 get_origin(const World &world) noexcept;
double get_origin(const World &world, int idx) noexcept;
World::Real3 get_spacing(const World &world) noexcept;
double get_spacing(const World &world, int idx) noexcept;

} // namespace pfc
