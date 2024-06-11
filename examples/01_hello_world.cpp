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

using namespace std;
using namespace pfc;

/** \example 01_hello_world.cpp
 *
 * Hello, World!
 *
 * The World class plays a vital role in OpenPFC by defining the parameters of
 * the calculation area. It sets the size of the area in the x, y, and z
 * directions and determines the origin's location.
 *
 * In OpenPFC, we establish the relationship between voxel indices (i, j, k) and
 * their corresponding spatial coordinates (x, y, z) with the following mapping:
 *
 *      x(i) = x0 + i * dx, where i ranges from 1 to Lx,
 *      y(j) = y0 + j * dy, where j ranges from 1 to Ly,
 *      z(k) = z0 + k * dz, where k ranges from 1 to Lz.
 *
 * It is important to note that the World class itself does not engage in
 * extensive computational tasks. Instead, it primarily serves as the
 * foundational definition of the calculation area. Typically, it represents one
 * of the initial components required when constructing a simulation model.
 *
 * This example demonstrates how to use the World class to create a simulation
 * world and retrieve some of its properties.
 */
int main() {
  // Create a world with custom dimensions and origin
  std::array<int, 3> dimensions = {10, 20, 30};
  std::array<double, 3> origin = {0.0, 0.0, 0.0};
  std::array<double, 3> discretization = {0.1, 0.1, 0.1};
  World world(dimensions, origin, discretization);

  // Retrieve world properties
  cout << "World properties:" << endl;
  cout << "Dimensions: " << world.get_Lx() << " x " << world.get_Ly() << " x " << world.get_Lz() << endl;
  cout << "Origin: (" << world.get_x0() << ", " << world.get_y0() << ", " << world.get_z0() << ")" << endl;
  cout << "Discretization: dx = " << world.get_dx() << ", dy = " << world.get_dy() << ", dz = " << world.get_dz()
       << endl;

  // Simpler way:
  cout << world << endl;

  // If just the size of the doman is defined, it is assumed that x0 = y0 = z0 =
  // 0 and dx = dy = dz = 1
  World world2({64, 64, 64});
  cout << world2 << endl;

  return 0;
}
