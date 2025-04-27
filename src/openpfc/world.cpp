// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/world.hpp"
#include <stdexcept>

namespace pfc {

World::World(const std::array<int, 3> &dimensions, const std::array<double, 3> &origin,
             const std::array<double, 3> &discretization)
    : Lx(dimensions[0]), Ly(dimensions[1]), Lz(dimensions[2]), x0(origin[0]), y0(origin[1]), z0(origin[2]),
      dx(discretization[0]), dy(discretization[1]), dz(discretization[2]) {

  if (Lx <= 0 || Ly <= 0 || Lz <= 0) {
    throw std::invalid_argument("Invalid dimensions. Lengths must be positive.");
  }

  if (dx <= 0 || dy <= 0 || dz <= 0) {
    throw std::invalid_argument("Invalid discretization. Values must be positive.");
  }
}

World::World(const std::array<int, 3> &dimensions) : World(dimensions, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}) {}

int World::get_Lx() const { return Lx; }
int World::get_Ly() const { return Ly; }
int World::get_Lz() const { return Lz; }
double World::get_x0() const { return x0; }
double World::get_y0() const { return y0; }
double World::get_z0() const { return z0; }
double World::get_dx() const { return dx; }
double World::get_dy() const { return dy; }
double World::get_dz() const { return dz; }

std::array<int, 3> World::get_size() const { return {Lx, Ly, Lz}; }
std::array<double, 3> World::get_origin() const { return {x0, y0, z0}; }
std::array<double, 3> World::get_discretization() const { return {dx, dy, dz}; }

World::operator heffte::box3d<int>() const { return heffte::box3d<int>({0, 0, 0}, {Lx - 1, Ly - 1, Lz - 1}); }

bool World::operator==(const World &other) const { return Lx == other.Lx && Ly == other.Ly && Lz == other.Lz; }

std::ostream &operator<<(std::ostream &os, const World &w) {
  os << "(Lx = " << w.Lx << ", Ly = " << w.Ly << ", Lz = " << w.Lz;
  os << ", x0 = " << w.x0 << ", y0 = " << w.y0 << ", z0 = " << w.z0;
  os << ", x1 = " << w.x0 + w.Lx * w.dx << ", y1 = " << w.y0 + w.Ly * w.dy << ", z0 = " << w.z0 + w.Lz * w.dz;
  os << ", dx = " << w.dx << ", dy = " << w.dy << ", dz = " << w.dz << ")";
  return os;
}

} // namespace pfc
