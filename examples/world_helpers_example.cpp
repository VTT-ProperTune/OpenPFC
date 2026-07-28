// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file world_helpers_example.cpp
 * @brief Example: Domain-first world construction
 *
 * Demonstrates the modern domain::create(Domain) API for creating World objects,
 * with clear separation between domain description (Domain) and world creation.
 *
 * The modern way to build a World:
 * 1. Describe the domain (bounds, spacing, MPI decomposition)
 * 2. Pass the Domain to domain::create(World)
 *
 * Legacy: create_world_uniform, create_world_from_bounds are superseded
 * by domain::create(Domain) and retained only for compatibility.
 */

#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/world.hpp>

using namespace pfc;

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::cout << "=== Domain-First World Construction Example ===\n\n";

  // ========================================================================
  // Example 1: Minimal domain creation
  // ========================================================================
  std::cout << "1. Minimal domain creation\n";
  std::cout << "----------------------------\n";

  // Describe domain: 64x64x64 cells, unit spacing, origin at (0,0,0)
  pfc::Domain domain1 = pfc::domain::create({64, 64, 64});

  // Create world from domain (full-grid subdomain for rank 0)
  pfc::Int3 lower1{0, 0, 0};
  pfc::Int3 upper1{63, 63, 63};
  pfc::World world1(lower1, upper1, domain1);

  std::cout << "  domain::create({64, 64, 64}):\n";
  std::cout << "    Domain: " << domain1 << "\n";
  std::cout << "    Rank " << rank << " local cells: "
            << (world1.subdomain_.high[0] - world1.subdomain_.low[0] + 1) *
               (world1.subdomain_.high[1] - world1.subdomain_.low[1] + 1) *
               (world1.subdomain_.high[2] - world1.subdomain_.low[2] + 1)
            << "\n\n";

  // ========================================================================
  // Example 2: Domain from discrete bounds
  // ========================================================================
  std::cout << "2. Domain from discrete bounds\n";
  std::cout << "--------------------------------\n";

  // Describe domain with explicit discrete bounds and spacing
  pfc::Int3 bounds_size{32, 32, 32};  // 32x32x32 cells
  pfc::Real3 spacing{1.0, 1.0, 1.0};
  pfc::Domain domain2 = pfc::domain::create(
      GridSize(bounds_size),
      PhysicalOrigin({0.0, 0.0, 0.0}),
      GridSpacing(spacing)
  );

  // Create world from this domain
  pfc::Int3 lower2{0, 0, 0};
  pfc::Int3 upper2{31, 31, 31};
  pfc::World world2(lower2, upper2, domain2);

  std::cout << "  Domain from discrete bounds (32³ cells, unit spacing):\n";
  std::cout << "    Domain: " << domain2 << "\n";
  std::cout << "    Physical volume: " << pfc::domain::physical_volume(domain2) << "\n\n";

  // ========================================================================
  // Example 3: Domain from physical bounds
  // ========================================================================
  std::cout << "3. Domain from physical bounds\n";
  std::cout << "-------------------------------\n";

  // Describe domain from physical bounds [0,0,0] to [10,10,10] with 100 cells
  pfc::Int3 size{100, 100, 100};
  pfc::Real3 lower{0.0, 0.0, 0.0};
  pfc::Real3 upper{10.0, 10.0, 10.0};
  pfc::Bool3 periodic{true, true, true};

  pfc::Domain domain3 = pfc::domain::from_bounds(size, lower, upper, periodic);

  std::cout << "  Domain from physical bounds (periodic):\n";
  std::cout << "    Size: " << size[0] << "x" << size[1] << "x" << size[2] << " cells\n";
  std::cout << "    Physical bounds: [" << lower[0] << "," << lower[1] << "," << lower[2]
            << "] to [" << upper[0] << "," << upper[1] << "," << upper[2] << "]\n";
  std::cout << "    Computed spacing: " << pfc::domain::get_spacing(domain3)[0] << "\n\n";

  // Non-periodic in x direction (different spacing formula)
  pfc::Bool3 periodic_x{false, true, true};
  pfc::Domain domain3_nonperiodic = pfc::domain::from_bounds(size, lower, upper, periodic_x);

  std::cout << "  Domain from physical bounds (non-periodic in x):\n";
  std::cout << "    x-spacing: " << std::fixed << std::setprecision(6)
            << pfc::domain::get_spacing(domain3_nonperiodic)[0]
            << " (vs " << (upper[0] - lower[0]) / (size[0] - 1) << " expected)\n\n";

  // ========================================================================
  // Example 4: Custom spacing and origin
  // ========================================================================
  std::cout << "4. Custom spacing and origin\n";
  std::cout << "------------------------------\n";

  // Custom spacing, default origin
  pfc::Int3 size4{64, 64, 128};
  pfc::Real3 spacing4{0.1, 0.1, 0.05};
  pfc::Domain domain4 = pfc::domain::with_spacing(size4, spacing4);

  std::cout << "  Domain with custom spacing:\n";
  std::cout << "    Size: " << size4[0] << "x" << size4[1] << "x" << size4[2] << "\n";
  std::cout << "    Spacing: [" << spacing4[0] << ", " << spacing4[1] << ", " << spacing4[2] << "]\n";
  std::cout << "    Domain: " << domain4 << "\n\n";

  // Custom origin, unit spacing
  pfc::Int3 size5{64, 64, 64};
  pfc::Real3 origin5{-5.0, -5.0, 0.0};
  pfc::Domain domain5 = pfc::domain::create(
      GridSize(size5),
      PhysicalOrigin(origin5),
      GridSpacing({1.0, 1.0, 1.0})
  );

  std::cout << "  Domain with custom origin:\n";
  std::cout << "    Size: " << size5[0] << "x" << size5[1] << "x" << size5[2] << "\n";
  std::cout << "    Origin: [" << origin5[0] << ", " << origin5[1] << ", " << origin5[2] << "]\n";
  std::cout << "    Domain: " << domain5 << "\n\n";

  // ========================================================================
  // Example 5: Strong types for type safety
  // ========================================================================
  std::cout << "5. Strong types for type safety\n";
  std::cout << "--------------------------------\n";

  // Using strong types prevents parameter order mistakes
  pfc::GridSize grid_size({256, 256, 256});
  pfc::PhysicalOrigin phys_origin({-128.0, -128.0, -128.0});
  pfc::GridSpacing grid_spacing({1.0, 1.0, 1.0});
  pfc::Bool3 periodicity{true, true, true};

  pfc::Domain domain6 = pfc::domain::create(grid_size, phys_origin, grid_spacing, periodicity);

  std::cout << "  Type-safe domain creation:\n";
  std::cout << "    GridSize: [" << grid_size.value[0] << ", " << grid_size.value[1]
            << ", " << grid_size.value[2] << "]\n";
  std::cout << "    PhysicalOrigin: [" << phys_origin.value[0] << ", " << phys_origin.value[1]
            << ", " << phys_origin.value[2] << "]\n";
  std::cout << "    GridSpacing: [" << grid_spacing.value[0] << ", " << grid_spacing.value[1]
            << ", " << grid_spacing.value[2] << "]\n";
  std::cout << "    Domain: " << domain6 << "\n\n";

  // ========================================================================
  // Example 6: Legacy API (for compatibility)
  // ========================================================================
  std::cout << "6. Legacy API (not recommended for new code)\n";
  std::cout << "--------------------------------------------\n";

  // Legacy: create_world_uniform(N) is superseded by domain::create(Domain).
  // Use Domain::from_discrete_bounds(bounds, spacing) + domain::create(world, domain).
  std::cout << "  Legacy: domain::create_world_uniform(64) still works\n";
  std::cout << "  Modern: domain::create({64, 64, 64}) is preferred\n\n";

  // Legacy: create_world_from_bounds is superseded by domain::from_bounds.
  // Use pfc::domain::from_bounds(size, lower, upper, periodic) directly.
  std::cout << "  Legacy: domain::create_world_from_bounds(...) still works\n";
  std::cout << "  Modern: pfc::domain::from_bounds(...) is preferred\n\n";

  // ========================================================================
  // Summary: Benefits of domain::create(Domain)
  // ========================================================================
  std::cout << "Benefits of Domain-First Construction:\n";
  std::cout << "======================================\n";
  std::cout << "  ✓ Clear separation: Domain describes space, World handles execution\n";
  std::cout << "  ✓ Type-safe: strong types prevent parameter order mistakes\n";
  std::cout << "  ✓ Flexible: reuse Domain objects across Worlds\n";
  std::cout << "  ✓ Explicit: all geometry parameters are clearly visible\n";
  std::cout << "  ✓ Zero overhead: Domain is an immutable value type\n";
  std::cout << "  ✓ Future-proof: ready for new decomposition strategies\n";

  MPI_Finalize();
  return 0;
}