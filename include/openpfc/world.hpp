#ifndef PFC_WORLD_HPP
#define PFC_WORLD_HPP

#include <array>
#include <heffte.h>
#include <iostream>

namespace pfc {
class World {
private:
  template <class T> using Vec3 = std::array<T, 3>;

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

  World(const Vec3<int> &dimensions, const Vec3<double> &origo,
        const Vec3<double> &discretization)
      : Lx(dimensions[0]), Ly(dimensions[1]), Lz(dimensions[2]), x0(origo[0]),
        y0(origo[1]), z0(origo[2]), dx(discretization[0]),
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

  World(const Vec3<int> &dimensions)
      : World(dimensions, Vec3<double>{0.0, 0.0, 0.0},
              Vec3<double>{1.0, 1.0, 1.0}) {}

  /**
   * @brief Get the size of the calculation domain.
   * @return The size of the domain: {Lx, Ly, Lz}.
   */
  Vec3<int> get_size() const { return Vec3<int>{Lx, Ly, Lz}; }

  /**
   * @brief Get the length in the x-direction.
   * @return The length in the x-direction.
   */
  int get_Lx() const { return Lx; }

  /**
   * @brief Get the length in the y-direction.
   * @return The length in the y-direction.
   */
  int get_Ly() const { return Ly; }

  /**
   * @brief Get the length in the z-direction.
   * @return The length in the z-direction.
   */
  int get_Lz() const { return Lz; }

  /**
   * @brief Get the origin coordinate in the x-direction.
   * @return The origin coordinate in the x-direction.
   */
  double get_x0() const { return x0; }

  /**
   * @brief Get the origin coordinate in the y-direction.
   * @return The origin coordinate in the y-direction.
   */
  double get_y0() const { return y0; }

  /**
   * @brief Get the origin coordinate in the z-direction.
   * @return The origin coordinate in the z-direction.
   */
  double get_z0() const { return z0; }

  /**
   * @brief Get the discretization parameter in the x-direction.
   * @return The discretization parameter in the x-direction.
   */
  double get_dx() const { return dx; }

  /**
   * @brief Get the discretization parameter in the y-direction.
   * @return The discretization parameter in the y-direction.
   */
  double get_dy() const { return dy; }

  /**
   * @brief Get the discretization parameter in the z-direction.
   * @return The discretization parameter in the z-direction.
   */
  double get_dz() const { return dz; }

  operator heffte::box3d<int>() const {
    return heffte::box3d<int>({0, 0, 0}, {Lx - 1, Ly - 1, Lz - 1});
  }

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
