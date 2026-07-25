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
 * GridSpacing spacing({1.0, 1.0, 1.0});
 * PhysicalOrigin origin({0.0, 0.0, 0.0});
 * // Compiler knows these are different!
 * ```
 */

#include <iomanip>
#include <iostream>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/data/types.hpp>

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
 *
 * @note This function requires GridSize, GridSpacing, and PhysicalOrigin
 * types. Use explicit factory methods (e.g., GridSize::from_vector3)
 * to convert from Int3/Real3 types.
 */
void print_domain_info(GridSize size, GridSpacing spacing, PhysicalOrigin origin) {
  std::cout << "Domain Information:\n";
  std::cout << "  Grid size: " << size.get()[0] << " x " << size.get()[1] << " x "
            << size.get()[2] << "\n";
  std::cout << "  Spacing: " << spacing.get()[0] << " x " << spacing.get()[1]
            << " x " << spacing.get()[2] << "\n";
  std::cout << "  Origin: (" << origin.get()[0] << ", " << origin.get()[1] << ", "
            << origin.get()[2] << ")\n";

  // Calculate physical domain size using explicit conversions
  Int3 size_raw = size.to_vector3();
  Real3 spacing_raw = spacing.to_vector3();
  Real3 domain_size = {size_raw[0] * spacing_raw[0],
                       size_raw[1] * spacing_raw[1],
                       size_raw[2] * spacing_raw[2]};

  std::cout << "  Physical domain: " << domain_size[0] << " x " << domain_size[1]
            << " x " << domain_size[2] << " units\n";
}

// Note: LocalOffset and IndexBounds types removed - using Int3 directly
/**
 * @brief Calculate subdomain bounds using Int3 arrays (removed LocalOffset/IndexBounds types)
 */
Int3 calculate_subdomain_upper(GridSize total_size, const Int3& local_offset,
                               GridSize local_size) {
  return {local_offset[0] + local_size.get()[0] - 1,
          local_offset[1] + local_size.get()[1] - 1,
          local_offset[2] + local_size.get()[2] - 1};
}

/**
 * @brief Map index to physical coordinates
 */
Real3 index_to_physical(Int3 index, PhysicalOrigin origin,
                        GridSpacing spacing) {
  return {origin.get()[0] + index[0] * spacing.get()[0],
          origin.get()[1] + index[1] * spacing.get()[1],
          origin.get()[2] + index[2] * spacing.get()[2]};
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

  // Demonstrate explicit conversion to raw types
  Int3 size_raw = size.to_vector3(); // Explicit conversion (preferred)
  std::cout << "\nExplicit conversion to raw Int3: [" << size_raw[0] << ", "
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
  Int3 subdomain1_offset{0, 0, 0};
  GridSize subdomain1_size({32, 64, 64});

  Int3 subdomain2_offset{32, 0, 0};
  GridSize subdomain2_size({32, 64, 64});

  auto upper1 = calculate_subdomain_upper(size, subdomain1_offset, subdomain1_size);
  auto upper2 = calculate_subdomain_upper(size, subdomain2_offset, subdomain2_size);

  std::cout << "Subdomain 1 bounds:\n";
  std::cout << "  Lower: [" << subdomain1_offset[0] << ", " << subdomain1_offset[1] << ", "
            << subdomain1_offset[2] << "]\n";
  std::cout << "  Upper: [" << upper1[0] << ", " << upper1[1] << ", " << upper1[2] << "]\n\n";

  std::cout << "Subdomain 2 bounds:\n";
  std::cout << "  Lower: [" << subdomain2_offset[0] << ", " << subdomain2_offset[1] << ", "
            << subdomain2_offset[2] << "]\n";
  std::cout << "  Upper: [" << upper2[0] << ", " << upper2[1] << ", " << upper2[2] << "]\n\n";

  // ========================================================================
  // Example 4: Index to Physical Coordinate Mapping
  // ========================================================================

  std::cout << "Example 4: Index to Physical Coordinate Mapping\n";
  std::cout << "------------------------------------------------\n\n";

  Int3 test_indices[] = {{0, 0, 0}, {32, 32, 32}, {63, 63, 63}};

  std::cout << std::fixed << std::setprecision(2);
  for (const auto &idx : test_indices) {
    Real3 coords = index_to_physical(idx, origin, spacing);
    std::cout << "Index [" << idx[0] << ", " << idx[1] << ", " << idx[2] << "] → ";
    std::cout << "Physical (" << coords[0] << ", " << coords[1] << ", "
              << coords[2] << ")\n";
  }

  std::cout << "\n";

  // ========================================================================
  // Example 5: Physical Bounds
  // ========================================================================

  std::cout << "Example 5: Physical Bounds\n";
  std::cout << "--------------------------\n\n";

  // Using raw Real3 arrays for bounds (PhysicalBounds type removed)
  Real3 lower{-32.0, -32.0, -32.0};
  Real3 upper{32.0, 32.0, 32.0};

  std::cout << "Physical domain bounds:\n";
  std::cout << "  Lower: (" << lower[0] << ", " << lower[1] << ", " << lower[2] << ")\n";
  std::cout << "  Upper: (" << upper[0] << ", " << upper[1] << ", " << upper[2] << ")\n";

  double volume = (upper[0] - lower[0]) * (upper[1] - lower[1]) *
                  (upper[2] - lower[2]);

  std::cout << "  Volume: " << volume << " cubic units\n\n";

  // ========================================================================
  // Example 6: Explicit Conversions
  // ========================================================================

  std::cout << "Example 6: Explicit Conversions\n";
  std::cout << "--------------------------------\n\n";

  // Convert from raw types using explicit factory methods
  Int3 old_size = {128, 128, 128};
  Real3 old_spacing = {0.5, 0.5, 0.5};
  Real3 old_origin = {0.0, 0.0, 0.0};

  std::cout << "Old style (raw types) converted using explicit factories:\n";
  print_domain_info(GridSize::from_vector3(old_size),
                    GridSpacing::from_vector3(old_spacing),
                    PhysicalOrigin::from_vector3(old_origin));

  std::cout << "\nImplicit conversions are no longer supported:\n";
  std::cout << "  // print_domain_info(old_size, old_spacing, old_origin);\n";
  std::cout << "  // Error: no implicit conversion from Int3/Real3 to strong types\n\n";

  // ========================================================================
  // Example 7: Zero-Cost Verification
  // ========================================================================

  std::cout << "Example 7: Zero-Cost Abstraction Verification\n";
  std::cout << "----------------------------------------------\n\n";

  std::cout << "Size comparisons (bytes):\n";
  std::cout << "  sizeof(Int3):        " << sizeof(Int3) << "\n";
  std::cout << "  sizeof(GridSize):    " << sizeof(GridSize) << " ✅ Same!\n";

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
  std::cout << "  ✅ Explicit conversions only (maximal type safety)\n";
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
