// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file world_helpers_example.cpp
 * @brief Example: Using Domain construction helper functions
 *
 * Demonstrates the modern Domain API for creating simulation domains
 * with less boilerplate and clearer intent using the pfc::domain namespace.
 */

#include <iomanip>
#include <iostream>
#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/data/world.hpp>

using namespace pfc;

int main() {
  std::cout << "=== Domain Construction Helpers Example ===\n\n";

  // ========================================================================
  // Example 1: Simple uniform grids
  // ========================================================================
  std::cout << "1. Uniform grids (most common case)\n";
  std::cout << "------------------------------------\n";

  // Create 64³ grid with unit spacing
  auto world1 = domain::create_world_uniform(64);
  std::cout << "  create_world_uniform(64):\n";
  auto size1 = world::get_size(world1);
  std::cout << "    Size: [" << size1[0] << ", " << size1[1] << ", " << size1[2]
            << "]\n";
  auto spacing1 = world::get_spacing(world1);
  std::cout << "    Spacing: [" << spacing1[0] << ", " << spacing1[1] << ", "
            << spacing1[2] << "]\n";
  auto origin1 = world::get_origin(world1);
  std::cout << "    Origin: [" << origin1[0] << ", " << origin1[1] << ", "
            << origin1[2] << "]\n\n";

  // Create 128³ grid with custom spacing
  auto world2 = domain::create_world_uniform(128, 0.5);
  std::cout << "  create_world_uniform(128, 0.5):\n";
  auto size2 = world::get_size(world2);
  std::cout << "    Size: [" << size2[0] << ", " << size2[1] << ", " << size2[2]
            << "]\n";
  auto spacing2 = world::get_spacing(world2);
  std::cout << "    Spacing: [" << spacing2[0] << ", " << spacing2[1] << ", "
            << spacing2[2] << "]\n\n";

  // ========================================================================
  // Example 2: Create from physical bounds
  // ========================================================================
  std::cout << "2. Create from physical bounds\n";
  std::cout << "-------------------------------\n";

  // Periodic domain from [0, 0, 0] to [10, 10, 10] with 100 cells
  auto world3 = domain::create_world_from_bounds({100, 100, 100},    // grid size
                                   {0.0, 0.0, 0.0},    // lower bounds
                                   {10.0, 10.0, 10.0}, // upper bounds
                                   {true, true, true}  // periodic in all directions
  );

  std::cout << "  create_world_from_bounds (periodic):\n";
  auto size3 = world::get_size(world3);
  std::cout << "    Size: [" << size3[0] << ", " << size3[1] << ", " << size3[2]
            << "]\n";
  auto spacing3 = world::get_spacing(world3);
  std::cout << "    Spacing: [" << spacing3[0] << ", " << spacing3[1] << ", "
            << spacing3[2] << "]\n";
  auto lower3 = world::get_lower(world3);
  std::cout << "    Lower: [" << lower3[0] << ", " << lower3[1] << ", " << lower3[2]
            << "]\n";
  auto upper3 = world::get_upper(world3);
  std::cout << "    Upper: [" << upper3[0] << ", " << upper3[1] << ", " << upper3[2]
            << "]\n";
  std::cout << "    (Spacing = (10-0)/100 = 0.1)\n\n";

  // Non-periodic in x direction
  auto world4 =
      domain::create_world_from_bounds({100, 100, 100}, {0.0, 0.0, 0.0}, {10.0, 10.0, 10.0},
                         {false, true, true} // non-periodic in x
      );

  std::cout << "  create_world_from_bounds (non-periodic in x):\n";
  auto spacing4 = world::get_spacing(world4);
  std::cout << "    Spacing: [" << spacing4[0] << ", " << spacing4[1] << ", "
            << spacing4[2] << "]\n";
  std::cout << "    (x-spacing = (10-0)/(100-1) = " << std::fixed
            << std::setprecision(6) << 10.0 / 99.0 << ")\n\n";

  // ========================================================================
  // Example 3: Custom spacing with default origin
  // ========================================================================
  std::cout << "3. Custom spacing, default origin\n";
  std::cout << "-----------------------------------\n";

  auto world5 = domain::create_world_with_spacing({64, 64, 128}, {0.1, 0.1, 0.05});
  std::cout << "  create_world_with_spacing({64, 64, 128}, {0.1, 0.1, 0.05}):\n";
  auto size5 = world::get_size(world5);
  std::cout << "    Size: [" << size5[0] << ", " << size5[1] << ", " << size5[2]
            << "]\n";
  auto spacing5 = world::get_spacing(world5);
  std::cout << "    Spacing: [" << spacing5[0] << ", " << spacing5[1] << ", "
            << spacing5[2] << "]\n";
  auto origin5 = world::get_origin(world5);
  std::cout << "    Origin: [" << origin5[0] << ", " << origin5[1] << ", "
            << origin5[2] << "]\n\n";

  // ========================================================================
  // Example 4: Custom origin with unit spacing
  // ========================================================================
  std::cout << "4. Custom origin, unit spacing\n";
  std::cout << "--------------------------------\n";

  auto world6 = domain::create_world_with_origin({64, 64, 64}, {-5.0, -5.0, 0.0});
  std::cout << "  create_world_with_origin({64, 64, 64}, {-5.0, -5.0, 0.0}):\n";
  auto size6 = world::get_size(world6);
  std::cout << "    Size: [" << size6[0] << ", " << size6[1] << ", " << size6[2]
            << "]\n";
  auto origin6 = world::get_origin(world6);
  std::cout << "    Origin: [" << origin6[0] << ", " << origin6[1] << ", "
            << origin6[2] << "]\n";
  auto spacing6 = world::get_spacing(world6);
  std::cout << "    Spacing: [" << spacing6[0] << ", " << spacing6[1] << ", "
            << spacing6[2] << "]\n\n";

  // ========================================================================
  // Example 5: Benefits summary
  // ========================================================================
  std::cout << "Benefits of Domain Helper Functions:\n";
  std::cout << "==============================\n";
  std::cout << "  ✓ Less boilerplate (create_world_uniform(64) vs create({64,64,64}, {0,0,0}, "
               "{1,1,1}))\n";
  std::cout << "  ✓ Self-documenting code (create_world_from_bounds shows intent)\n";
  std::cout << "  ✓ Automatic spacing calculation (no manual math)\n";
  std::cout << "  ✓ Early validation (catch errors at construction)\n";
  std::cout << "  ✓ Zero runtime overhead (all inline)\n";
  std::cout << "  ✓ Modern API (part of pfc::domain namespace)\n";

  return 0;
}