/*

OpenPFC, a simulation software for the phase field crystal method.
Copyright (C) 2024 VTT Technical Research Centre of Finland Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.

*/

#include <iostream>
#include <openpfc/world.hpp>

using namespace pfc;
using namespace std;

int main() {
  // World constructor:
  // 1. number of grid points in x, y, z direction (Lx, Ly, Lz)
  // 2. origo of the domain (x0, y0, z0)
  // 3. discretization parameter in each direction (dx, dy, dz)

  // Thus, the mapping (i, j, k) -> (x, y, z) is
  // x(i) = x0 + i*dx, i = {1, 2, 3, ..., Lx}
  // y(j) = y0 + j*dy, j = {1, 2, 3, ..., Ly}
  // z(k) = z0 + k*dz, k = {1, 2, 3, ..., Lz}

  World w({128, 128, 128}, {-64.0, -64.0, -64.0}, {1.0, 1.0, 1.0});
  cout << w << endl;

  // If just the size of the doman is defined, it is assumed that x0 = y0 = z0 =
  // 0 and dx = dy = dz = 1
  World w2({64, 64, 64});
  cout << w2 << endl;
  return 0;
}
