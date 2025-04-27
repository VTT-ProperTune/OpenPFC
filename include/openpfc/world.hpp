// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef PFC_WORLD_HPP
#define PFC_WORLD_HPP

#include <array>
#include <heffte.h>
#include <iostream>

namespace pfc {

/**
 * @brief Represents a world in the simulation domain.
 *
 * The World class encapsulates the dimensions, origin coordinates, and
 * discretization parameters of a simulation world. It provides accessors to
 * retrieve the properties of the world and supports conversion to
 * heffte::box3d<int> for interoperability with the HeFFTe library.
 *
 * Example usage:
 * @code
 * // Create a world with dimensions (100, 100, 100)
 * World world({100, 100, 100});
 *
 * // Print out some information about world
 * std::cout << world << std::endl;
 * @endcode
 */
class World {
private:
public:
  // TODO: to be moved to private later on
  const int Lx;    ///< Length in the x-direction
  const int Ly;    ///< Length in the y-direction
  const int Lz;    ///< Length in the z-direction
  const double x0; ///< Origin coordinate in the x-direction
  const double y0; ///< Origin coordinate in the y-direction
  const double z0; ///< Origin coordinate in the z-direction
  const double dx; ///< Discretization parameter in the x-direction
  const double dy; ///< Discretization parameter in the y-direction
  const double dz; ///< Discretization parameter in the z-direction

  /**
   * @brief Constructs a World object with the specified dimensions, origin, and
   * discretization.
   *
   * @param dimensions The dimensions of the world in the form {Lx, Ly, Lz}.
   * @param origin The origin coordinates of the world in the form {x0, y0, z0}.
   * @param discretization The discretization parameters of the world in the
   * form {dx, dy, dz}.
   *
   * @throws std::invalid_argument if any of the dimensions or discretization
   * values are non-positive.
   */
  World(const std::array<int, 3> &dimensions, const std::array<double, 3> &origin,
        const std::array<double, 3> &discretization);

  /**
   * @brief Constructs a World object with the specified dimensions and default
   * origin and discretization.
   *
   * @param dimensions The dimensions of the world in the form {Lx, Ly, Lz}.
   *
   * @throws std::invalid_argument if any of the dimensions are non-positive.
   */
  World(const std::array<int, 3> &dimensions);

  // Getters for member variables
  int get_Lx() const;
  int get_Ly() const;
  int get_Lz() const;
  double get_x0() const;
  double get_y0() const;
  double get_z0() const;
  double get_dx() const;
  double get_dy() const;
  double get_dz() const;

  /**
   * @brief Get the size of the calculation domain.
   * @return The size of the domain: {Lx, Ly, Lz}.
   */
  std::array<int, 3> get_size() const;

  /**
   * @brief Get the origin of the coordinate system
   *
   * @return std::array<double, 3>
   */
  std::array<double, 3> get_origin() const;

  /**
   * @brief Get the discretization of the coordinate system
   *
   * @return std::array<double, 3>
   */
  std::array<double, 3> get_discretization() const;

  /**
   * @brief Conversion operator to heffte::box3d<int>.
   *
   * Allows implicit conversion of a World object to heffte::box3d<int>.
   * The resulting box represents the entire world domain.
   *
   * @return A heffte::box3d<int> representing the world domain.
   */
  operator heffte::box3d<int>() const;

  /**
   * @brief Compare this world to other world.
   *
   * @param other world
   * @return true
   * @return false
   */
  bool operator==(const World &other) const;

  /**
   * @brief Output stream operator for World objects.
   *
   * Allows printing the state of a World object to an output stream.
   *
   * @param os The output stream to write to.
   * @param w The World object to be printed.
   * @return The updated output stream.
   */
  friend std::ostream &operator<<(std::ostream &os, const World &w);
};

} // namespace pfc

#endif // PFC_WORLD_HPP
