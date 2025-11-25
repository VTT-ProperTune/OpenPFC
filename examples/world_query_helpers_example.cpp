// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file world_query_helpers_example.cpp
 * @brief Example: Using World query convenience functions
 *
 * Demonstrates the new helper functions for querying World properties:
 * - physical_volume(): Calculate domain volume
 * - is_1d/is_2d/is_3d(): Check dimensionality
 * - dimensionality(): Get dimensionality as integer
 * - get_lower_bounds/get_upper_bounds(): Domain bounds
 */

#include <iomanip>
#include <iostream>
#include <openpfc/core/world.hpp>

using namespace pfc;

void print_world_info(const World &world, const std::string &name) {
  std::cout << "\n" << name << ":\n";
  std::cout << std::string(60, '=') << "\n";

  // Size information
  auto size = world::get_size(world);
  std::cout << "  Grid size:        " << size[0] << " × " << size[1] << " × "
            << size[2] << "\n";
  std::cout << "  Total points:     " << world::get_total_size(world) << "\n";

  // Spacing and origin
  auto spacing = world::get_spacing(world);
  auto origin = world::get_origin(world);
  std::cout << "  Spacing:          (" << spacing[0] << ", " << spacing[1] << ", "
            << spacing[2] << ")\n";
  std::cout << "  Origin:           (" << origin[0] << ", " << origin[1] << ", "
            << origin[2] << ")\n";

  // NEW: Physical volume
  std::cout << "\n  Physical volume:  " << std::fixed << std::setprecision(6)
            << world::physical_volume(world) << "\n";

  // NEW: Dimensionality checks
  std::cout << "  Dimensionality:   " << world::dimensionality(world) << "D\n";
  std::cout << "    is_1d():        " << (world::is_1d(world) ? "true" : "false")
            << "\n";
  std::cout << "    is_2d():        " << (world::is_2d(world) ? "true" : "false")
            << "\n";
  std::cout << "    is_3d():        " << (world::is_3d(world) ? "true" : "false")
            << "\n";

  // NEW: Domain bounds
  auto lower = world::get_lower_bounds(world);
  auto upper = world::get_upper_bounds(world);
  std::cout << "\n  Lower bounds:     (" << lower[0] << ", " << lower[1] << ", "
            << lower[2] << ")\n";
  std::cout << "  Upper bounds:     (" << upper[0] << ", " << upper[1] << ", "
            << upper[2] << ")\n";

  // Calculate physical extents
  std::cout << "  Physical extent:  (" << upper[0] - lower[0] << " × "
            << upper[1] - lower[1] << " × " << upper[2] - lower[2] << ")\n";
}

int main() {
  std::cout << "=============================================================\n";
  std::cout << "OpenPFC Example: World Query Convenience Functions\n";
  std::cout << "=============================================================\n";

  // Example 1: 3D cubic domain
  std::cout << "\nExample 1: Standard 3D simulation domain\n";
  auto world3d = world::create({64, 64, 64}, {0.0, 0.0, 0.0}, {0.1, 0.1, 0.1});
  print_world_info(world3d, "3D Cubic Domain (64³, dx=0.1)");

  // Example 2: 2D domain (quasi-2D simulation)
  std::cout << "\n\nExample 2: 2D simulation (thin film)\n";
  auto world2d = world::create({128, 128, 1}, {0.0, 0.0, 0.0}, {0.01, 0.01, 1.0});
  print_world_info(world2d, "2D Domain (128² × 1, dx=0.01)");

  // Example 3: 1D domain
  std::cout << "\n\nExample 3: 1D simulation (line)\n";
  auto world1d = world::create({256, 1, 1}, {0.0, 0.0, 0.0}, {0.05, 1.0, 1.0});
  print_world_info(world1d, "1D Domain (256 × 1 × 1, dx=0.05)");

  // Example 4: Non-cubic 3D domain with offset origin
  std::cout << "\n\nExample 4: Non-cubic domain with custom origin\n";
  auto world_offset =
      world::create({100, 100, 50}, {-5.0, -5.0, 0.0}, {0.1, 0.1, 0.2});
  print_world_info(world_offset, "Offset Domain (100×100×50, origin at (-5,-5,0))");

  // Example 5: Using convenience functions without world:: prefix (ADL)
  std::cout << "\n\nExample 5: Using ADL (no namespace prefix needed)\n";
  std::cout << std::string(60, '=') << "\n";
  {
    using namespace world; // Enable ADL

    auto w = create({32, 32, 32}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});

    // All these work without world:: prefix via ADL!
    std::cout << "  Volume:           " << physical_volume(w) << "\n";
    std::cout << "  Is 3D:            " << (is_3d(w) ? "yes" : "no") << "\n";
    std::cout << "  Dimensionality:   " << dimensionality(w) << "D\n";

    auto lower = get_lower_bounds(w);
    auto upper = get_upper_bounds(w);
    std::cout << "  Bounds:           [" << lower[0] << ", " << upper[0] << "] × "
              << "[" << lower[1] << ", " << upper[1] << "] × "
              << "[" << lower[2] << ", " << upper[2] << "]\n";
  }

  // Example 6: Comparing manual calculation with convenience function
  std::cout << "\n\nExample 6: Manual vs. convenience function\n";
  std::cout << std::string(60, '=') << "\n";
  {
    auto w = world::create({50, 50, 50}, {0.0, 0.0, 0.0}, {0.2, 0.2, 0.2});

    // Manual calculation
    auto spacing = world::get_spacing(w);
    auto size = world::get_size(w);
    double manual_vol =
        spacing[0] * spacing[1] * spacing[2] * size[0] * size[1] * size[2];

    // Using convenience function
    double conv_vol = world::physical_volume(w);

    std::cout << "  Manual calculation:   " << std::fixed << std::setprecision(6)
              << manual_vol << "\n";
    std::cout << "  Convenience function: " << conv_vol << "\n";
    std::cout << "  Match:                "
              << (std::abs(manual_vol - conv_vol) < 1e-10 ? "✓" : "✗") << "\n";
  }

  std::cout << "\n\n=============================================================\n";
  std::cout << "Summary:\n";
  std::cout << "=============================================================\n";
  std::cout << "New convenience functions added to pfc::world namespace:\n\n";
  std::cout << "  physical_volume(world)     - Calculate domain volume\n";
  std::cout << "  is_1d(world)               - Check if 1D (nx>1, ny=1, nz=1)\n";
  std::cout << "  is_2d(world)               - Check if 2D (nx>1, ny>1, nz=1)\n";
  std::cout << "  is_3d(world)               - Check if 3D (all > 1)\n";
  std::cout << "  dimensionality(world)      - Get 1, 2, or 3\n";
  std::cout << "  get_lower_bounds(world)    - Physical coords at (0,0,0)\n";
  std::cout << "  get_upper_bounds(world)    - Physical coords at max indices\n";
  std::cout << "\nThese functions:\n";
  std::cout << "  ✓ Are inline (zero runtime cost)\n";
  std::cout << "  ✓ Are noexcept (no exceptions)\n";
  std::cout << "  ✓ Work via ADL (no namespace prefix needed)\n";
  std::cout << "  ✓ Improve code readability\n";
  std::cout << "  ✓ Follow OpenPFC's functional design philosophy\n";
  std::cout << "=============================================================\n\n";

  return 0;
}
