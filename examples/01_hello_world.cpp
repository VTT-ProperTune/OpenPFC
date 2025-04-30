// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>
#include <openpfc/core/world.hpp>

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
  World world = create_world(dimensions, origin, discretization);

  // Retrieve world properties
  cout << "World properties:" << endl;
  auto size = get_size(world);
  cout << "Dimensions: " << size[0] << " x " << size[1] << " x " << size[2] << endl;
  auto world_origin = get_origin(world);
  cout << "Origin: (" << world_origin[0] << ", " << world_origin[1] << ", " << world_origin[2] << ")" << endl;
  auto spacing = get_spacing(world);
  cout << "Spacing: dx = " << spacing[0] << ", dy = " << spacing[1] << ", dz = " << spacing[2] << endl;

  // Simpler way:
  cout << world << endl;

  // If just the size of the doman is defined, it is assumed that x0 = y0 = z0 =
  // 0 and dx = dy = dz = 1
  World world2 = create_world({64, 64, 64});
  cout << world2 << endl;

  return 0;
}
