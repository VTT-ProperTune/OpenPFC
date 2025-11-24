// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file 16_strong_types_demo.cpp
 * @brief Demonstration of strong type aliases for geometric quantities
 *
 * This example shows how strong types make code more self-documenting
 * and catch argument order mistakes at compile time.
 *
 * ## What You'll Learn
 *
 * - How to use strong types (GridSize, GridSpacing, PhysicalOrigin)
 * - Benefits of type safety in function signatures
 * - Zero-cost abstraction (compiles to same assembly as raw types)
 * - Backward compatibility with existing code
 *
 * ## Key Concepts
 *
 * **Problem**: Raw Int3 and Real3 types don't distinguish between
 * different kinds of geometric quantities:
 *
 * ```cpp
 * Int3 size = {64, 64, 64};
 * Int3 offset = {0, 0, 0};
 * // Easy to confuse! Both are just Int3
 * ```
 *
 * **Solution**: Strong types make the distinction explicit:
 *
 * ```cpp
 * GridSize size({64, 64, 64});
 * LocalOffset offset({0, 0, 0});
 * // Compiler knows these are different!
 * ```
 */

#include <iomanip>
#include <iostream>
#include <openpfc/core/strong_types.hpp>
#include <openpfc/core/types.hpp>

using namespace pfc;

// ============================================================================
// Example Functions Using Strong Types
// ============================================================================

/**
 * @brief Print domain information using strong types
 *
 * Function signature is self-documenting - no need to check docs
 * to understand what each parameter means.
 *
 * @param size Grid dimensions
 * @param spacing Physical spacing between grid points
 * @param origin Physical origin of coordinate system
 */
void print_domain_info(GridSize size, GridSpacing spacing, PhysicalOrigin origin) {
  std::cout << "Domain Information:\n";
  std::cout << "  Grid size: " << size.get()[0] << " x " << size.get()[1] << " x "
            << size.get()[2] << "\n";
  std::cout << "  Spacing: " << spacing.get()[0] << " x " << spacing.get()[1]
            << " x " << spacing.get()[2] << "\n";
  std::cout << "  Origin: (" << origin.get()[0] << ", " << origin.get()[1] << ", "
            << origin.get()[2] << ")\n";

  // Calculate physical domain size
  Real3 domain_size = {size.get()[0] * spacing.get()[0],
                       size.get()[1] * spacing.get()[1],
                       size.get()[2] * spacing.get()[2]};

  std::cout << "  Physical domain: " << domain_size[0] << " x " << domain_size[1]
            << " x " << domain_size[2] << " units\n";
}

/**
 * @brief Calculate subdomain bounds using strong types
 *
 * This function demonstrates type safety - you cannot accidentally
 * pass spacing where offset is expected.
 */
IndexBounds calculate_subdomain_bounds(GridSize total_size, LocalOffset local_offset,
                                       GridSize local_size) {
  Int3 lower = local_offset.get();
  Int3 upper = {local_offset.get()[0] + local_size.get()[0] - 1,
                local_offset.get()[1] + local_size.get()[1] - 1,
                local_offset.get()[2] + local_size.get()[2] - 1};

  return IndexBounds(lower, upper);
}

/**
 * @brief Map index to physical coordinates
 */
PhysicalCoords index_to_physical(Int3 index, PhysicalOrigin origin,
                                 GridSpacing spacing) {
  Real3 coords = {origin.get()[0] + index[0] * spacing.get()[0],
                  origin.get()[1] + index[1] * spacing.get()[1],
                  origin.get()[2] + index[2] * spacing.get()[2]};

  return PhysicalCoords(coords);
}

// ============================================================================
// Main Example
// ============================================================================

int main() {
  std::cout << "=============================================================\n";
  std::cout << "OpenPFC Strong Types Demonstration\n";
  std::cout << "=============================================================\n\n";

  // ========================================================================
  // Example 1: Basic Strong Type Usage
  // ========================================================================

  std::cout << "Example 1: Basic Strong Type Usage\n";
  std::cout << "-----------------------------------\n\n";

  // Create strong types with clear intent
  GridSize size({64, 64, 64});
  GridSpacing spacing({1.0, 1.0, 1.0});
  PhysicalOrigin origin({-32.0, -32.0, -32.0});

  std::cout << "Created domain with strong types:\n";
  print_domain_info(size, spacing, origin);

  // Demonstrate implicit conversion to raw types
  Int3 size_raw = size; // Implicit conversion
  std::cout << "\nImplicit conversion to raw Int3: [" << size_raw[0] << ", "
            << size_raw[1] << ", " << size_raw[2] << "]\n";

  std::cout << "\n";

  // ========================================================================
  // Example 2: Type Safety Benefits
  // ========================================================================

  std::cout << "Example 2: Type Safety Benefits\n";
  std::cout << "--------------------------------\n\n";

  std::cout << "✅ Correct function call:\n";
  print_domain_info(size, spacing, origin);

  std::cout << "\n❌ The following would NOT compile:\n";
  std::cout << "   print_domain_info(spacing, size, origin);\n";
  std::cout << "   // Error: cannot convert GridSpacing to GridSize\n\n";

  // ========================================================================
  // Example 3: Subdomain Calculations
  // ========================================================================

  std::cout << "Example 3: Subdomain Calculations\n";
  std::cout << "----------------------------------\n\n";

  // Define subdomains (as in domain decomposition)
  LocalOffset subdomain1_offset({0, 0, 0});
  GridSize subdomain1_size({32, 64, 64});

  LocalOffset subdomain2_offset({32, 0, 0});
  GridSize subdomain2_size({32, 64, 64});

  auto bounds1 =
      calculate_subdomain_bounds(size, subdomain1_offset, subdomain1_size);
  auto bounds2 =
      calculate_subdomain_bounds(size, subdomain2_offset, subdomain2_size);

  std::cout << "Subdomain 1 bounds:\n";
  std::cout << "  Lower: [" << bounds1.lower[0] << ", " << bounds1.lower[1] << ", "
            << bounds1.lower[2] << "]\n";
  std::cout << "  Upper: [" << bounds1.upper[0] << ", " << bounds1.upper[1] << ", "
            << bounds1.upper[2] << "]\n\n";

  std::cout << "Subdomain 2 bounds:\n";
  std::cout << "  Lower: [" << bounds2.lower[0] << ", " << bounds2.lower[1] << ", "
            << bounds2.lower[2] << "]\n";
  std::cout << "  Upper: [" << bounds2.upper[0] << ", " << bounds2.upper[1] << ", "
            << bounds2.upper[2] << "]\n\n";

  // ========================================================================
  // Example 4: Index to Physical Coordinate Mapping
  // ========================================================================

  std::cout << "Example 4: Index to Physical Coordinate Mapping\n";
  std::cout << "------------------------------------------------\n\n";

  Int3 test_indices[] = {{0, 0, 0}, {32, 32, 32}, {63, 63, 63}};

  std::cout << std::fixed << std::setprecision(2);
  for (const auto &idx : test_indices) {
    PhysicalCoords coords = index_to_physical(idx, origin, spacing);
    std::cout << "Index [" << idx[0] << ", " << idx[1] << ", " << idx[2] << "] → ";
    std::cout << "Physical (" << coords.get()[0] << ", " << coords.get()[1] << ", "
              << coords.get()[2] << ")\n";
  }

  std::cout << "\n";

  // ========================================================================
  // Example 5: Physical Bounds
  // ========================================================================

  std::cout << "Example 5: Physical Bounds\n";
  std::cout << "--------------------------\n\n";

  PhysicalBounds domain_bounds({-32.0, -32.0, -32.0}, {32.0, 32.0, 32.0});

  std::cout << "Physical domain bounds:\n";
  std::cout << "  Lower: (" << domain_bounds.lower[0] << ", "
            << domain_bounds.lower[1] << ", " << domain_bounds.lower[2] << ")\n";
  std::cout << "  Upper: (" << domain_bounds.upper[0] << ", "
            << domain_bounds.upper[1] << ", " << domain_bounds.upper[2] << ")\n";

  double volume = (domain_bounds.upper[0] - domain_bounds.lower[0]) *
                  (domain_bounds.upper[1] - domain_bounds.lower[1]) *
                  (domain_bounds.upper[2] - domain_bounds.lower[2]);

  std::cout << "  Volume: " << volume << " cubic units\n\n";

  // ========================================================================
  // Example 6: Backward Compatibility
  // ========================================================================

  std::cout << "Example 6: Backward Compatibility\n";
  std::cout << "----------------------------------\n\n";

  // Old style code still works
  Int3 old_size = {128, 128, 128};
  Real3 old_spacing = {0.5, 0.5, 0.5};
  Real3 old_origin = {0.0, 0.0, 0.0};

  std::cout << "Old style (raw types) works:\n";
  print_domain_info(old_size, old_spacing, old_origin); // Implicit conversion!

  std::cout << "\n";

  // ========================================================================
  // Example 7: Zero-Cost Verification
  // ========================================================================

  std::cout << "Example 7: Zero-Cost Abstraction Verification\n";
  std::cout << "----------------------------------------------\n\n";

  std::cout << "Size comparisons (bytes):\n";
  std::cout << "  sizeof(Int3):        " << sizeof(Int3) << "\n";
  std::cout << "  sizeof(GridSize):    " << sizeof(GridSize) << " ✅ Same!\n";
  std::cout << "  sizeof(LocalOffset): " << sizeof(LocalOffset) << " ✅ Same!\n\n";

  std::cout << "  sizeof(Real3):          " << sizeof(Real3) << "\n";
  std::cout << "  sizeof(GridSpacing):    " << sizeof(GridSpacing) << " ✅ Same!\n";
  std::cout << "  sizeof(PhysicalOrigin): " << sizeof(PhysicalOrigin)
            << " ✅ Same!\n\n";

  std::cout << "Memory layout properties:\n";
  std::cout << "  GridSize is trivially copyable:  "
            << (std::is_trivially_copyable_v<GridSize> ? "✅ Yes" : "❌ No") << "\n";
  std::cout << "  GridSize has standard layout:    "
            << (std::is_standard_layout_v<GridSize> ? "✅ Yes" : "❌ No") << "\n";
  std::cout << "  GridSpacing is trivially copyable: "
            << (std::is_trivially_copyable_v<GridSpacing> ? "✅ Yes" : "❌ No")
            << "\n";
  std::cout << "  GridSpacing has standard layout:   "
            << (std::is_standard_layout_v<GridSpacing> ? "✅ Yes" : "❌ No") << "\n";

  // ========================================================================
  // Summary
  // ========================================================================

  std::cout << "\n=============================================================\n";
  std::cout << "Summary\n";
  std::cout << "=============================================================\n\n";

  std::cout << "Strong types provide:\n";
  std::cout << "  ✅ Type safety (compiler catches mistakes)\n";
  std::cout << "  ✅ Self-documenting code (clear intent)\n";
  std::cout << "  ✅ Zero runtime cost (same size, same performance)\n";
  std::cout << "  ✅ Backward compatibility (implicit conversions)\n";
  std::cout << "  ✅ Better IDE support (autocomplete knows types)\n\n";

  std::cout << "Use strong types for:\n";
  std::cout << "  • Function parameters (clarity)\n";
  std::cout << "  • Public APIs (self-documenting)\n";
  std::cout << "  • Struct members (semantic meaning)\n\n";

  std::cout << "Raw types are fine for:\n";
  std::cout << "  • Local variables (less ceremony)\n";
  std::cout << "  • Tight loops (no difference anyway)\n";
  std::cout << "  • Internal helpers (context is clear)\n\n";

  return 0;
}
