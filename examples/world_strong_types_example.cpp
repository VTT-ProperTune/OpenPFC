// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @example world_strong_types_example.cpp
 * @brief Type-safe Domain construction using strong types
 *
 * Strong types (GridSize, PhysicalOrigin, GridSpacing) make
 * `domain::create` self-documenting and reject swapped arguments at
 * compile time.
 *
 * @code
 * auto domain = domain::create(
 *     GridSize({256, 256, 256}),
 *     PhysicalOrigin({0, 0, 0}),
 *     GridSpacing({1, 1, 1})
 * );
 * @endcode
 */

#include <iostream>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/strong_types.hpp>

int main() {
  using namespace pfc;
  using namespace pfc::domain;

  std::cout << "OpenPFC Strong Types Example\n";
  std::cout << "=============================\n\n";

  std::cout << "1. Basic Domain creation with strong types:\n";
  {
    GridSize size({64, 64, 64});
    PhysicalOrigin origin({-32.0, -32.0, -32.0});
    GridSpacing spacing({1.0, 1.0, 1.0});

    auto domain = create(size, origin, spacing);

    std::cout << "   Grid size: " << get_size(domain)[0] << "³\n";
    std::cout << "   Physical origin: (" << get_origin(domain)[0] << ", "
              << get_origin(domain)[1] << ", " << get_origin(domain)[2] << ")\n";
    std::cout << "   Grid spacing: " << get_spacing(domain)[0] << "\n";
  }
  std::cout << "\n";

  std::cout << "2. Inline construction:\n";
  {
    auto domain =
        create(GridSize({128, 128, 128}), PhysicalOrigin({-64.0, -64.0, -64.0}),
               GridSpacing({0.5, 0.5, 0.5}));

    std::cout << "   Created 128³ grid with spacing 0.5\n";
    std::cout << "   Domain extends from " << get_origin(domain)[0] << " to ";
    Real3 upper_corner = to_coords(domain, get_size(domain));
    std::cout << upper_corner[0] << "\n";
  }
  std::cout << "\n";

  std::cout << "3. Non-uniform grid (different sizes and spacing):\n";
  {
    auto domain = create(GridSize({256, 128, 64}), PhysicalOrigin({0.0, 0.0, 0.0}),
                         GridSpacing({0.1, 0.2, 0.4}));

    std::cout << "   Grid: " << get_size(domain)[0] << "×" << get_size(domain)[1]
              << "×" << get_size(domain)[2] << "\n";
    std::cout << "   Spacing: dx=" << get_spacing(domain)[0]
              << ", dy=" << get_spacing(domain)[1]
              << ", dz=" << get_spacing(domain)[2] << "\n";

    Real3 size_phys = {get_size(domain)[0] * get_spacing(domain)[0],
                       get_size(domain)[1] * get_spacing(domain)[1],
                       get_size(domain)[2] * get_spacing(domain)[2]};
    std::cout << "   Physical size: " << size_phys[0] << "×" << size_phys[1] << "×"
              << size_phys[2] << "\n";
  }
  std::cout << "\n";

  std::cout << "4. Type safety demonstration:\n";
  {
    GridSize size({64, 64, 64});
    PhysicalOrigin origin({0.0, 0.0, 0.0});
    GridSpacing spacing({1.0, 1.0, 1.0});
    auto domain1 = create(size, origin, spacing);
    (void)domain1;
    std::cout << "   ✓ Correct: create(size, origin, spacing)\n";
    // auto bad1 = create(spacing, size, origin);  // Compile error!
    std::cout << "   ✗ Wrong parameter orders rejected at compile time\n";
  }
  std::cout << "\n";

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

  std::cout << "6. Converting from raw types to strong types:\n";
  {
    Int3 size = {32, 32, 32};
    Real3 offset = {0.0, 0.0, 0.0};
    Real3 spacing = {1.0, 1.0, 1.0};
    auto domain =
        create(GridSize::from_vector3(size), PhysicalOrigin::from_vector3(offset),
               GridSpacing::from_vector3(spacing));
    (void)domain;
    std::cout << "   Raw types converted to strong types!\n";
    std::cout << "   Primary API: domain::create() returns Domain\n";
  }
  std::cout << "\n";

  std::cout << "7. Helper factories:\n";
  {
    auto d1 = create(Int3{64, 64, 64});
    std::cout << "   create({64,64,64}) creates 64³ grid\n";
    auto d2 = with_spacing({128, 128, 128}, {0.5, 0.5, 0.5});
    std::cout << "   with_spacing(128³, 0.5) creates 128³ grid with spacing 0.5\n";
    auto d3 = from_bounds({100, 100, 100}, {0, 0, 0}, {10, 10, 10});
    std::cout << "   from_bounds() computes spacing automatically\n";
    (void)d1;
    (void)d2;
    (void)d3;
  }
  std::cout << "\n";

  std::cout << "8. Coordinate transformations:\n";
  {
    auto domain =
        create(GridSize({64, 64, 64}), PhysicalOrigin({-32.0, -32.0, -32.0}),
               GridSpacing({1.0, 1.0, 1.0}));

    Real3 center = to_coords(domain, {32, 32, 32});
    std::cout << "   Center index (32,32,32) maps to physical (" << center[0] << ","
              << center[1] << "," << center[2] << ")\n";

    Real3 corner = to_coords(domain, {0, 0, 0});
    std::cout << "   Origin index (0,0,0) maps to physical (" << corner[0] << ","
              << corner[1] << "," << corner[2] << ")\n";
  }
  std::cout << "\n";

  std::cout << "Strong types make code safer and more readable!\n";

  return 0;
}
