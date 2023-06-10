#include <heffte.h>
#include <iostream>
#include <openpfc/world.hpp>

/** \example world_example.cpp
 *
 * This example demonstrates how to use the World class to create a simulation
 * world, retrieve its properties, and convert it to a heffte::box3d<int>.
 */
int main() {
  // Create a world with custom dimensions and origin
  pfc::World world({10, 20, 30}, {0.0, 0.0, 0.0}, {0.1, 0.1, 0.1});

  // Retrieve world properties
  std::cout << "World properties:" << std::endl;
  std::cout << "Dimensions: " << world.get_Lx() << " x " << world.get_Ly()
            << " x " << world.get_Lz() << std::endl;
  std::cout << "Origin: (" << world.get_x0() << ", " << world.get_y0() << ", "
            << world.get_z0() << ")" << std::endl;
  std::cout << "Discretization: dx = " << world.get_dx()
            << ", dy = " << world.get_dy() << ", dz = " << world.get_dz()
            << std::endl;

  // Convert to heffte::box3d<int>
  heffte::box3d<int> box = static_cast<heffte::box3d<int>>(world);
  std::cout << "Box dimensions: " << box.size[0] << " x " << box.size[1]
            << " x " << box.size[2] << std::endl;

  return 0;
}
