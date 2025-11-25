// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @example world_strong_types_example.cpp
 * @brief Demonstrates type-safe World construction using strong types
 *
 * This example shows how to use strong types (GridSize, PhysicalOrigin, GridSpacing)
 * for clear, self-documenting, and type-safe World creation. Strong types prevent
 * parameter confusion at compile time.
 *
 * ## Why Strong Types?
 *
 * **Before (ambiguous):**
 * @code
 * auto world = world::create({256, 256, 256}, {0, 0, 0}, {1, 1, 1});
 * // Which is which? Have to check documentation!
 * @endcode
 *
 * **After (self-documenting):**
 * @code
 * auto world = world::create(
 *     GridSize({256, 256, 256}),      // Clear: grid dimensions
 *     PhysicalOrigin({0, 0, 0}),      // Clear: origin location
 *     GridSpacing({1, 1, 1})          // Clear: grid spacing
 * );
 * // Impossible to swap parameters!
 * @endcode
 */

#include <iomanip>
#include <iostream>
#include <openpfc/core/strong_types.hpp>
#include <openpfc/core/world.hpp>

int main() {
  using namespace pfc;
  using namespace pfc::world;

  std::cout << "OpenPFC Strong Types Example\n";
  std::cout << "=============================\n\n";

  // ========================================================================
  // Example 1: Basic usage with strong types
  // ========================================================================
  std::cout << "1. Basic World creation with strong types:\n";
  {
    GridSize size({64, 64, 64});
    PhysicalOrigin origin({-32.0, -32.0, -32.0});
    GridSpacing spacing({1.0, 1.0, 1.0});

    auto world = create(size, origin, spacing);

    std::cout << "   Grid size: " << get_size(world)[0] << "³\n";
    std::cout << "   Physical origin: (" << get_origin(world)[0] << ", "
              << get_origin(world)[1] << ", " << get_origin(world)[2] << ")\n";
    std::cout << "   Grid spacing: " << get_spacing(world)[0] << "\n";
  }
  std::cout << "\n";

  // ========================================================================
  // Example 2: Inline construction (most concise)
  // ========================================================================
  std::cout << "2. Inline construction:\n";
  {
    auto world =
        create(GridSize({128, 128, 128}), PhysicalOrigin({-64.0, -64.0, -64.0}),
               GridSpacing({0.5, 0.5, 0.5}));

    std::cout << "   Created 128³ grid with spacing 0.5\n";
    std::cout << "   Domain extends from " << get_origin(world)[0] << " to ";

    // Calculate upper bound
    Real3 upper_corner = to_coords(world, get_size(world));
    std::cout << upper_corner[0] << "\n";
  }
  std::cout << "\n";

  // ========================================================================
  // Example 3: Non-uniform grid
  // ========================================================================
  std::cout << "3. Non-uniform grid (different sizes and spacing):\n";
  {
    GridSize size({256, 128, 64});
    PhysicalOrigin origin({0.0, 0.0, 0.0});
    GridSpacing spacing({0.1, 0.2, 0.4});

    auto world = create(size, origin, spacing);

    std::cout << "   Grid: " << get_size(world)[0] << "×" << get_size(world)[1]
              << "×" << get_size(world)[2] << "\n";
    std::cout << "   Spacing: dx=" << get_spacing(world)[0]
              << ", dy=" << get_spacing(world)[1] << ", dz=" << get_spacing(world)[2]
              << "\n";

    // Physical domain size
    Real3 size_phys = {get_size(world)[0] * get_spacing(world)[0],
                       get_size(world)[1] * get_spacing(world)[1],
                       get_size(world)[2] * get_spacing(world)[2]};
    std::cout << "   Physical size: " << size_phys[0] << "×" << size_phys[1] << "×"
              << size_phys[2] << "\n";
  }
  std::cout << "\n";

  // ========================================================================
  // Example 4: Type safety - prevents parameter confusion
  // ========================================================================
  std::cout << "4. Type safety demonstration:\n";
  {
    GridSize size({64, 64, 64});
    PhysicalOrigin origin({0.0, 0.0, 0.0});
    GridSpacing spacing({1.0, 1.0, 1.0});

    // This compiles - correct order
    auto world1 = create(size, origin, spacing);
    std::cout << "   ✓ Correct: create(size, origin, spacing)\n";

    // These would NOT compile (uncomment to verify):
    // auto bad1 = create(spacing, size, origin);  // Compile error!
    // auto bad2 = create(origin, spacing, size);  // Compile error!
    // auto bad3 = create(size, spacing, origin);  // Compile error!
    std::cout << "   ✗ Wrong parameter orders rejected at compile time\n";
  }
  std::cout << "\n";

  // ========================================================================
  // Example 5: Zero overhead - same performance as raw types
  // ========================================================================
  std::cout << "5. Zero overhead verification:\n";
  {
    std::cout << "   sizeof(GridSize) == sizeof(Int3): "
              << (sizeof(GridSize) == sizeof(Int3) ? "✓" : "✗") << "\n";
    std::cout << "   sizeof(PhysicalOrigin) == sizeof(Real3): "
              << (sizeof(PhysicalOrigin) == sizeof(Real3) ? "✓" : "✗") << "\n";
    std::cout << "   sizeof(GridSpacing) == sizeof(Real3): "
              << (sizeof(GridSpacing) == sizeof(Real3) ? "✓" : "✗") << "\n";
    std::cout << "   Strong types compile away completely!\n";
  }
  std::cout << "\n";

  // ========================================================================
  // Example 6: Backward compatibility
  // ========================================================================
  std::cout << "6. Backward compatibility with raw types:\n";
  {
    // Old API still works (though deprecated)
    Int3 size = {32, 32, 32};
    Real3 offset = {0.0, 0.0, 0.0};
    Real3 spacing = {1.0, 1.0, 1.0};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    auto world = create(size, offset, spacing);
#pragma GCC diagnostic pop

    std::cout << "   Old API still works (with deprecation warning)\n";
    std::cout << "   Migration: Just wrap parameters in strong types!\n";
  }
  std::cout << "\n";

  // ========================================================================
  // Example 7: Working with World helper functions
  // ========================================================================
  std::cout << "7. Helper functions use strong types internally:\n";
  {
    // uniform() now uses strong types internally
    auto world1 = uniform(64);
    std::cout << "   uniform(64) creates 64³ grid\n";

    auto world2 = uniform(128, 0.5);
    std::cout << "   uniform(128, 0.5) creates 128³ grid with spacing 0.5\n";

    auto world3 = from_bounds({100, 100, 100}, {0, 0, 0}, {10, 10, 10});
    std::cout << "   from_bounds() computes spacing automatically\n";
  }
  std::cout << "\n";

  // ========================================================================
  // Example 8: Coordinate transformations
  // ========================================================================
  std::cout << "8. Coordinate transformations:\n";
  {
    auto world =
        create(GridSize({64, 64, 64}), PhysicalOrigin({-32.0, -32.0, -32.0}),
               GridSpacing({1.0, 1.0, 1.0}));

    // Index to physical coordinates
    Real3 center = to_coords(world, {32, 32, 32});
    std::cout << "   Center index (32,32,32) maps to physical (" << center[0] << ","
              << center[1] << "," << center[2] << ")\n";

    // Origin corner
    Real3 corner = to_coords(world, {0, 0, 0});
    std::cout << "   Origin index (0,0,0) maps to physical (" << corner[0] << ","
              << corner[1] << "," << corner[2] << ")\n";
  }
  std::cout << "\n";

  std::cout << "Strong types make code safer and more readable!\n";

  return 0;
}
