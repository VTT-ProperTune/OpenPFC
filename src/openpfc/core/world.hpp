// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

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
  Int3 m_size;     ///< Dimensions of the world: {Lx, Ly, Lz}
  Real3 m_origin;  ///< Origin coordinates: {x0, y0, z0}
  Real3 m_spacing; ///< Spacing parameters: {dx, dy, dz}

public:
  /**
   * @brief Constructs a World object with the specified dimensions, origin, and
   * spacing.
   *
   * @param dimensions The dimensions of the world in the form {Lx, Ly, Lz}.
   * @param origin The origin coordinates of the world in the form {x0, y0, z0}.
   * @param spacing The spacing parameters of the world in the form {dx, dy, dz}.
   *
   * @throws std::invalid_argument if any of the dimensions or spacing
   * values are non-positive.
   */
  World(const Int3 &dimensions, const Real3 &origin, const Real3 &spacing);

  /**
   * @brief Constructs a World object with the specified dimensions and default
   * origin and spacing.
   *
   * Default origin is {0.0, 0.0, 0.0}, and default spacing is {1.0, 1.0, 1.0}.
   *
   * @param dimensions The dimensions of the world in the form {Lx, Ly, Lz}.
   *
   * @throws std::invalid_argument if any of the dimensions are non-positive.
   */
  World(const Int3 &dimensions);

  // Getters for member variables
  Int3 get_size() const noexcept;
  Real3 get_origin() const noexcept;
  Real3 get_spacing() const noexcept;

  /**
   * @brief Get the size of the calculation domain.
   * @return The size of the domain: {Lx, Ly, Lz}.
   */
  Int3 size() const noexcept;

  /**
   * @brief Get the origin of the coordinate system
   *
   * @return Real3
   */
  Real3 origin() const noexcept;

  /**
   * @brief Get the spacing of the coordinate system
   *
   * @return Real3
   */
  Real3 spacing() const noexcept;

  /**
   * @brief Get the number of grid points in each dimension.
   *
   * @return The number of grid points in each dimension: {Nx, Ny, Nz}.
   */
  int total_size() const noexcept;

  /**
   * @brief Computes the physical coordinate corresponding to grid indices {i,j,k}.
   *
   * This method calculates the physical coordinates in the simulation domain
   * based on the grid indices and spacing.
   *
   * @param indices The grid indices {i, j, k}.
   * @return The physical coordinate {x, y, z}.
   */
  Real3 physical_coordinates(const Int3 &indices) const noexcept;

  /**
   * @brief Computes the grid indices corresponding to physical coordinates {x,y,z}.
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
World create_world(const World::Int3 &dimensions, const World::Real3 &origin, const World::Real3 &spacing) noexcept;

/**
 * @brief Create a World object with the specified dimensions and default
 * origin and spacing.
 *
 * Default origin is {0.0, 0.0, 0.0}, and default spacing is {1.0, 1.0, 1.0}.
 */
World create_world(const World::Int3 &dimensions) noexcept;

} // namespace pfc
