// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file box3d.hpp
 * @brief 3D integer bounding box for grid index space
 *
 * @details
 * Defines the Box3D class which represents a discrete 3D rectangular region
 * in grid index space. This is used to describe local computational domains
 * in parallel simulations.
 *
 * Box3D operates purely in index space and is independent of physical
 * coordinates, spacing, or origin. It is typically paired with World
 * for coordinate transformations.
 *
 * @code
 * #include <openpfc/core/box3d.hpp>
 *
 * // Create a box from (0,0,0) to (63,63,63)
 * pfc::Box3D box({0, 0, 0}, {64, 64, 64});
 * auto size = box.size();  // Returns {64, 64, 64}
 * @endcode
 *
 * @see core/world.hpp for physical coordinate mapping
 * @see core/decomposition.hpp for domain decomposition using Box3D
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef PFC_BOX3D_HPP
#define PFC_BOX3D_HPP

#include <array>
#include <ostream>
#include <stdexcept>

namespace pfc {

/**
 * @brief Represents a 3D integer box in grid index space.
 *
 * The Box3D class describes a discrete 3D rectangular region
 * using integer lower and upper corner indices.
 *
 * It is designed to represent local computational domains in the simulation,
 * distinct from the continuous physical domain described by World.
 *
 * Box3D is purely an index-space concept:
 * it does not know about physical coordinates, spacing, or origin.
 *
 * Example usage:
 * @code
 * #include "openpfc/core/box3d.hpp"
 * #include <iostream>
 *
 * int main() {
 *     pfc::Box3D box({0, 0, 0}, {9, 9, 9});
 *     std::cout << "Box size: {"
 *               << box.size()[0] << ", "
 *               << box.size()[1] << ", "
 *               << box.size()[2] << "}" << std::endl;
 * }
 * @endcode
 *
 * Design Responsibilities:
 * - Define a 3D discrete region.
 * - Support size calculation and basic box queries.
 * - Serve as a basis for parallel decomposition and FFT layouts.
 *
 * This class follows the Single Responsibility Principle (SRP).
 */
class Box3D {
public:
  /// Type aliases for clarity
  using Int3 = std::array<int, 3>;

private:
  Int3 m_lower; ///< Lower corner indices: {i_min, j_min, k_min}
  Int3 m_upper; ///< Upper corner indices: {i_max, j_max, k_max}

public:
  /**
   * @brief Constructs a Box3D with given lower and upper corners.
   *
   * @param lower Lower corner indices {i_min, j_min, k_min}.
   * @param upper Upper corner indices {i_max, j_max, k_max}.
   *
   * @throws std::invalid_argument if lower > upper in any dimension.
   */
  Box3D(const Int3 &lower, const Int3 &upper);

  /**
   * @brief Returns the lower corner indices.
   * @return const Int3& Lower corner {i_min, j_min, k_min}.
   */
  const Int3 &lower() const noexcept;

  /**
   * @brief Returns the upper corner indices.
   * @return const Int3& Upper corner {i_max, j_max, k_max}.
   */
  const Int3 &upper() const noexcept;

  /**
   * @brief Returns the size (number of elements) in each dimension.
   *
   * Size is computed as (upper - lower + 1) per dimension.
   *
   * @return Int3 Size {Nx, Ny, Nz}.
   */
  Int3 size() const noexcept;

  /**
   * @brief Computes the total number of grid points in the box.
   *
   * Equivalent to size()[0] * size()[1] * size()[2].
   *
   * @return int Total number of grid points.
   */
  int total_size() const noexcept;

  /**
   * @brief Check if a given index {i,j,k} is inside this box.
   *
   * @param index A grid index {i, j, k}.
   * @return true if the index is inside the box, false otherwise.
   */
  bool contains(const Int3 &index) const noexcept;

  /**
   * @brief Equality operator.
   *
   * @param other Another box to compare.
   * @return true if both lower and upper corners are identical.
   */
  bool operator==(const Box3D &other) const noexcept;

  /**
   * @brief Inequality operator.
   *
   * @param other Another box to compare.
   * @return true if either lower or upper corners differ.
   */
  bool operator!=(const Box3D &other) const noexcept;

  /**
   * @brief Output stream operator for Box3D objects.
   *
   * Allows easy printing of Box3D state for debugging and logging.
   *
   * @param os The output stream.
   * @param box The Box3D object.
   * @return The updated output stream.
   */
  friend std::ostream &operator<<(std::ostream &os, const Box3D &box) noexcept;
};

} // namespace pfc

#endif // PFC_BOX3D_HPP
