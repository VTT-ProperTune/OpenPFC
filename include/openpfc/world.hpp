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
  World(const std::array<int, 3> &dimensions,
        const std::array<double, 3> &origin,
        const std::array<double, 3> &discretization)
      : Lx(dimensions[0]), Ly(dimensions[1]), Lz(dimensions[2]), x0(origin[0]),
        y0(origin[1]), z0(origin[2]), dx(discretization[0]),
        dy(discretization[1]), dz(discretization[2]) {

    // Validate dimensions
    if (Lx <= 0 || Ly <= 0 || Lz <= 0) {
      throw std::invalid_argument(
          "Invalid dimensions. Lengths must be positive.");
    }

    // Validate discretization
    if (dx <= 0 || dy <= 0 || dz <= 0) {
      throw std::invalid_argument(
          "Invalid discretization. Values must be positive.");
    }
  }

  /**
   * @brief Constructs a World object with the specified dimensions and default
   * origin and discretization.
   *
   * @param dimensions The dimensions of the world in the form {Lx, Ly, Lz}.
   *
   * @throws std::invalid_argument if any of the dimensions are non-positive.
   */
  World(const std::array<int, 3> &dimensions)
      : World(dimensions, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}) {}

  // Getters for member variables
  int get_Lx() const { return Lx; }
  int get_Ly() const { return Ly; }
  int get_Lz() const { return Lz; }
  double get_x0() const { return x0; }
  double get_y0() const { return y0; }
  double get_z0() const { return z0; }
  double get_dx() const { return dx; }
  double get_dy() const { return dy; }
  double get_dz() const { return dz; }

  /**
   * @brief Get the size of the calculation domain.
   * @return The size of the domain: {Lx, Ly, Lz}.
   */
  std::array<int, 3> get_size() const { return {Lx, Ly, Lz}; }

  /**
   * @brief Get the origin of the coordinate system
   *
   * @return std::array<double, 3>
   */
  std::array<double, 3> get_origin() const { return {x0, y0, z0}; }

  /**
   * @brief Get the discretization of the coordinate system
   *
   * @return std::array<double, 3>
   */
  std::array<double, 3> get_discretization() const { return {dx, dy, dz}; }

  /**
   * @brief Conversion operator to heffte::box3d<int>.
   *
   * Allows implicit conversion of a World object to heffte::box3d<int>.
   * The resulting box represents the entire world domain.
   *
   * @return A heffte::box3d<int> representing the world domain.
   */
  operator heffte::box3d<int>() const {
    return heffte::box3d<int>({0, 0, 0}, {Lx - 1, Ly - 1, Lz - 1});
  }

  /**
   * @brief Compare this world to other world.
   *
   * @param other world
   * @return true
   * @return false
   */
  bool operator==(const World& other) const {
    return Lx == other.Lx &&
           Ly == other.Ly &&
           Lz == other.Lz;
  }

  /**
   * @brief Output stream operator for World objects.
   *
   * Allows printing the state of a World object to an output stream.
   *
   * @param os The output stream to write to.
   * @param w The World object to be printed.
   * @return The updated output stream.
   */
  friend std::ostream &operator<<(std::ostream &os, const World &w) {
    os << "(Lx = " << w.Lx << ", Ly = " << w.Ly << ", Lz = " << w.Lz;
    os << ", x0 = " << w.x0 << ", y0 = " << w.y0 << ", z0 = " << w.z0;
    os << ", x1 = " << w.x0 + w.Lx * w.dx << ", y1 = " << w.y0 + w.Ly * w.dy
       << ", z0 = " << w.z0 + w.Lz * w.dz;
    os << ", dx = " << w.dx << ", dy = " << w.dy << ", dz = " << w.dz << ")";
    return os;
  };
};

} // namespace pfc

#endif // PFC_WORLD_HPP
