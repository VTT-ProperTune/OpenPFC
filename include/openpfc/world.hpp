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
  const int Lx, Ly, Lz;
  const double x0, y0, z0;
  const double dx, dy, dz;

  World(const Vec3<int> &dimensions, const Vec3<double> &origo,
        const Vec3<double> &discretization)
      : Lx(dimensions[0]), Ly(dimensions[1]), Lz(dimensions[2]), x0(origo[0]),
        y0(origo[1]), z0(origo[2]), dx(discretization[0]),
        dy(discretization[1]), dz(discretization[2]) {}

  World(const Vec3<int> &dimensions)
      : World(dimensions, Vec3<double>{0.0, 0.0, 0.0},
              Vec3<double>{1.0, 1.0, 1.0}) {}

  Vec3<int> get_size() const { return Vec3<int>{Lx, Ly, Lz}; }

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
