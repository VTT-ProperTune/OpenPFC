// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

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
