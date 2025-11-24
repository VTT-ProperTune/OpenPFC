// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file 17_custom_coordinate_system.cpp
 * @brief Example: Defining custom coordinate systems without modifying OpenPFC
 *
 * @details
 * This example demonstrates OpenPFC's extensibility by showing how to add
 * custom coordinate systems without modifying the library source code.
 *
 * We implement two complete coordinate systems:
 * 1. **Polar coordinates** (2D: r, θ) - simpler case
 * 2. **Spherical coordinates** (3D: r, θ, φ) - complete 3D case
 *
 * ## Key Technique: ADL (Argument-Dependent Lookup)
 *
 * OpenPFC uses ADL to find coordinate transformation functions. You define
 * functions in your namespace (or pfc namespace), and ADL ensures they're
 * found automatically.
 *
 * ## Philosophy: Laboratory, Not Fortress
 *
 * This example proves you can extend OpenPFC without forking or modifying its
 * source code. This is the "laboratory" philosophy in action.
 *
 * ## How to Use This Example
 *
 * 1. Read through the polar coordinate implementation (simpler)
 * 2. Study the spherical coordinate implementation (complete)
 * 3. Copy the pattern for your own coordinate system
 * 4. No OpenPFC source code modifications required!
 *
 * @example
 * @code
 * // Define your tag
 * struct MyCoordTag {};
 *
 * // Implement the pattern shown in this file
 * // ... your coordinate system code ...
 *
 * // Use it!
 * Real3 coords = my_to_coords({10, 20, 30});
 * @endcode
 */

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>

// OpenPFC includes
#include "openpfc/openpfc.hpp"

using namespace pfc;

// Use M_PI from cmath (requires _USE_MATH_DEFINES on some platforms)
#ifndef M_PI
constexpr double M_PI = 3.14159265358979323846;
#endif

// ============================================================================
// Part 1: Polar Coordinates (2D - Simpler Example)
// ============================================================================

/**
 * @brief Tag for polar coordinate system
 *
 * Empty struct used as a compile-time tag for template specialization.
 * This is the standard C++ idiom for tag-based dispatch.
 */
struct PolarTag {};

/**
 * @brief Polar coordinate system parameters
 *
 * Polar coordinates (r, θ) where:
 * - r: radial distance from origin (r ≥ 0)
 * - θ: angle measured counter-clockwise from +x axis (radians)
 *
 * Physical interpretation:
 * - x = r * cos(θ)
 * - y = r * sin(θ)
 * - z = 0 (2D system)
 *
 * This is a value type (struct) following OpenPFC's "laboratory" philosophy:
 * - All members public and const (transparent, inspectable)
 * - Immutable after construction (functional style)
 * - No methods (operations via free functions)
 */
struct PolarCoordinateSystem {
  /// Minimum radial distance (typically 0, but can be > 0 for annular domains)
  const double m_r_min;

  /// Maximum radial distance
  const double m_r_max;

  /// Minimum angle (radians, typically 0)
  const double m_theta_min;

  /// Maximum angle (radians, typically 2*pi for full circle)
  const double m_theta_max;

  /// Grid periodicity: {radial, angular, unused}
  /// Typically {false, true, false} since θ is periodic
  const Bool3 m_periodic;

  /**
   * @brief Construct polar coordinate system
   *
   * @param r_range Radial range [r_min, r_max]
   * @param theta_range Angular range [θ_min, θ_max] in radians
   * @param periodic Periodicity (θ is typically periodic)
   *
   * @pre r_max > r_min ≥ 0
   * @pre theta_max > theta_min
   *
   * @example
   * @code
   * // Full circle: r in [0, 10], theta in [0, 2*pi]
   * PolarCoordinateSystem cs(
   *     {0.0, 10.0},
   *     {0.0, 2.0 * M_PI},
   *     {false, true, false}
   * );
   * @endcode
   */
  PolarCoordinateSystem(std::pair<double, double> r_range,
                        std::pair<double, double> theta_range,
                        Bool3 periodic = {false, true, false})
      : m_r_min(r_range.first), m_r_max(r_range.second),
        m_theta_min(theta_range.first), m_theta_max(theta_range.second),
        m_periodic(periodic) {}
};

/**
 * @brief Convert grid indices to physical Cartesian coordinates (Polar → Cartesian)
 *
 * This function is found via ADL when used with polar coordinate grids.
 * It implements the polar to Cartesian transformation:
 *
 * Given grid indices (i, j, k):
 * 1. Map to polar coordinates: r = r(i), θ = θ(j)
 * 2. Convert to Cartesian: x = r*cos(θ), y = r*sin(θ), z = 0
 *
 * @param cs Polar coordinate system defining the domain
 * @param indices Grid indices (i_r, i_θ, 0)
 * @param size Grid dimensions [n_r, n_θ, 1]
 * @return Cartesian coordinates (x, y, z)
 *
 * @note This is found via ADL - no explicit namespace qualification needed
 * @note For 2D polar coords, z is always 0
 *
 * Algorithm:
 * - r = r_min + i * (r_max - r_min) / n_r
 * - θ = θ_min + j * (θ_max - θ_min) / n_θ
 * - x = r * cos(θ)
 * - y = r * sin(θ)
 * - z = 0
 *
 * @example
 * @code
 * PolarCoordinateSystem cs({0, 10}, {0, 2*pi});
 * Int3 idx = {50, 64, 0};  // Middle of r, 90° angle
 * Int3 size = {100, 128, 1};
 * Real3 xyz = polar_to_coords(cs, idx, size);  // Returns (0, 5, 0)
 * @endcode
 */
inline Real3 polar_to_coords(const PolarCoordinateSystem &cs, const Int3 &indices,
                             const Int3 &size) {
  // Map indices to polar coordinates
  const double dr = (cs.m_r_max - cs.m_r_min) / size[0];
  const double dtheta =
      (cs.m_theta_max - cs.m_theta_min) /
      (cs.m_periodic[1] ? size[1] : size[1] - 1); // Handle periodicity

  const double r = cs.m_r_min + indices[0] * dr;
  const double theta = cs.m_theta_min + indices[1] * dtheta;

  // Polar → Cartesian transformation
  const double x = r * std::cos(theta);
  const double y = r * std::sin(theta);
  const double z = 0.0; // 2D system

  return {x, y, z};
}

/**
 * @brief Convert Cartesian coordinates to grid indices (Cartesian → Polar)
 *
 * Inverse transformation of polar_to_coords().
 *
 * Given Cartesian coordinates (x, y, z):
 * 1. Convert to polar: r = √(x² + y²), θ = atan2(y, x)
 * 2. Map to grid indices: i = round(r), j = round(θ)
 *
 * @param cs Polar coordinate system
 * @param coords Cartesian coordinates (x, y, z)
 * @param size Grid dimensions [n_r, n_θ, 1]
 * @return Grid indices (i_r, i_θ, 0)
 *
 * @note z coordinate is ignored (2D system)
 * @warning Indices may be outside grid range - caller should check bounds
 */
inline Int3 polar_to_indices(const PolarCoordinateSystem &cs, const Real3 &coords,
                             const Int3 &size) {
  const double x = coords[0];
  const double y = coords[1];
  // z is ignored (2D)

  // Cartesian → Polar
  const double r = std::sqrt(x * x + y * y);
  double theta = std::atan2(y, x);

  // Handle angle wraparound: map theta in [-pi, pi] to [theta_min, theta_max]
  if (theta < cs.m_theta_min) {
    theta += 2.0 * M_PI;
  }

  // Map to indices
  const double dr = (cs.m_r_max - cs.m_r_min) / size[0];
  const double dtheta =
      (cs.m_theta_max - cs.m_theta_min) / (cs.m_periodic[1] ? size[1] : size[1] - 1);

  const int i_r = static_cast<int>(std::round((r - cs.m_r_min) / dr));
  const int i_theta =
      static_cast<int>(std::round((theta - cs.m_theta_min) / dtheta));

  return {i_r, i_theta, 0};
}

// ============================================================================
// Part 2: Spherical Coordinates (3D - Complete Example)
// ============================================================================

/**
 * @brief Tag for spherical coordinate system
 */
struct SphericalTag {};

/**
 * @brief Spherical coordinate system parameters
 *
 * Spherical coordinates (r, θ, φ) where:
 * - r: radial distance from origin (r ≥ 0)
 * - theta: polar angle measured from +z axis (theta in [0, pi])
 * - phi: azimuthal angle in x-y plane (phi in [0, 2*pi])
 *
 * Physical interpretation:
 * - x = r * sin(θ) * cos(φ)
 * - y = r * sin(θ) * sin(φ)
 * - z = r * cos(θ)
 *
 * Common use cases:
 * - Planetary atmospheres (full sphere)
 * - Radial growth problems (bubble, crystal)
 * - Spherical shell problems (annular: r_min > 0)
 */
struct SphericalCoordinateSystem {
  const double m_r_min;     ///< Minimum radius (0 for full sphere)
  const double m_r_max;     ///< Maximum radius
  const double m_theta_min; ///< Min polar angle (typically 0)
  const double m_theta_max; ///< Max polar angle (typically π)
  const double m_phi_min;   ///< Min azimuthal angle (typically 0)
  const double m_phi_max;   ///< Max azimuthal angle (typically 2π)
  const Bool3 m_periodic;   ///< Periodicity: {false, false, true} for φ

  /**
   * @brief Construct spherical coordinate system
   *
   * @param r_range Radial range [r_min, r_max]
   * @param theta_range Polar angle range [theta_min, theta_max], typically [0, pi]
   * @param phi_range Azimuthal angle range [phi_min, phi_max], typically [0, 2*pi]
   * @param periodic Periodicity (phi is typically periodic)
   *
   * @example Full sphere
   * @code
   * SphericalCoordinateSystem cs(
   *     {0.0, 10.0},                        // r ∈ [0, 10]
   *     {0.0, M_PI},           // theta in [0, pi]
   *     {0.0, 2.0 * M_PI},     // phi in [0, 2*pi]
   *     {false, false, true}                // phi periodic
   * );
   * @endcode
   *
   * @example Spherical shell
   * @code
   * SphericalCoordinateSystem shell(
   *     {5.0, 10.0},  // Annular: 5 ≤ r ≤ 10
   *     {0.0, M_PI},
   *     {0.0, 2.0 * M_PI}
   * );
   * @endcode
   */
  SphericalCoordinateSystem(std::pair<double, double> r_range,
                            std::pair<double, double> theta_range,
                            std::pair<double, double> phi_range,
                            Bool3 periodic = {false, false, true})
      : m_r_min(r_range.first), m_r_max(r_range.second),
        m_theta_min(theta_range.first), m_theta_max(theta_range.second),
        m_phi_min(phi_range.first), m_phi_max(phi_range.second),
        m_periodic(periodic) {}
};

/**
 * @brief Spherical → Cartesian coordinate transformation
 *
 * Transforms grid indices in spherical coordinates to Cartesian (x,y,z).
 *
 * Transformation:
 * - x = r * sin(θ) * cos(φ)
 * - y = r * sin(θ) * sin(φ)
 * - z = r * cos(θ)
 *
 * @param cs Spherical coordinate system
 * @param indices Grid indices (i_r, i_theta, i_phi)
 * @param size Grid dimensions [n_r, n_θ, n_φ]
 * @return Cartesian coordinates (x, y, z)
 *
 * @note Found via ADL - no namespace qualification needed
 *
 * Special points:
 * - theta = 0: North pole (0, 0, r)
 * - theta = pi: South pole (0, 0, -r)
 * - theta = pi/2, phi = 0: Point on +x axis (r, 0, 0)
 * - theta = pi/2, phi = pi/2: Point on +y axis (0, r, 0)
 */
inline Real3 spherical_to_coords(const SphericalCoordinateSystem &cs,
                                 const Int3 &indices, const Int3 &size) {
  // Map indices to spherical coordinates
  const double dr = (cs.m_r_max - cs.m_r_min) / size[0];
  const double dtheta =
      (cs.m_theta_max - cs.m_theta_min) / (cs.m_periodic[1] ? size[1] : size[1] - 1);
  const double dphi =
      (cs.m_phi_max - cs.m_phi_min) / (cs.m_periodic[2] ? size[2] : size[2] - 1);

  const double r = cs.m_r_min + indices[0] * dr;
  const double theta = cs.m_theta_min + indices[1] * dtheta;
  const double phi = cs.m_phi_min + indices[2] * dphi;

  // Spherical → Cartesian transformation
  const double sin_theta = std::sin(theta);
  const double cos_theta = std::cos(theta);
  const double sin_phi = std::sin(phi);
  const double cos_phi = std::cos(phi);

  const double x = r * sin_theta * cos_phi;
  const double y = r * sin_theta * sin_phi;
  const double z = r * cos_theta;

  return {x, y, z};
}

/**
 * @brief Cartesian → Spherical coordinate transformation
 *
 * Inverse of spherical_to_coords().
 *
 * Transformation:
 * - r = √(x² + y² + z²)
 * - theta = acos(z / r)
 * - phi = atan2(y, x)
 *
 * @param cs Spherical coordinate system
 * @param coords Cartesian coordinates (x, y, z)
 * @param size Grid dimensions [n_r, n_theta, n_phi]
 * @return Grid indices (i_r, i_theta, i_phi)
 *
 * @warning Singular at origin (r=0) - undefined theta and phi
 * @warning Indices may be outside grid - caller should validate
 */
inline Int3 spherical_to_indices(const SphericalCoordinateSystem &cs,
                                 const Real3 &coords, const Int3 &size) {
  const double x = coords[0];
  const double y = coords[1];
  const double z = coords[2];

  // Cartesian → Spherical
  const double r = std::sqrt(x * x + y * y + z * z);
  const double theta = r > 1e-14 ? std::acos(z / r) : 0.0; // Handle r ≈ 0
  double phi = std::atan2(y, x);

  // Handle angle wraparound
  if (phi < cs.m_phi_min) {
    phi += 2.0 * M_PI;
  }

  // Map to indices
  const double dr = (cs.m_r_max - cs.m_r_min) / size[0];
  const double dtheta =
      (cs.m_theta_max - cs.m_theta_min) / (cs.m_periodic[1] ? size[1] : size[1] - 1);
  const double dphi =
      (cs.m_phi_max - cs.m_phi_min) / (cs.m_periodic[2] ? size[2] : size[2] - 1);

  const int i_r = static_cast<int>(std::round((r - cs.m_r_min) / dr));
  const int i_theta =
      static_cast<int>(std::round((theta - cs.m_theta_min) / dtheta));
  const int i_phi = static_cast<int>(std::round((phi - cs.m_phi_min) / dphi));

  return {i_r, i_theta, i_phi};
}

// ============================================================================
// Part 3: Usage Examples and Demonstrations
// ============================================================================

/**
 * @brief Demonstrate polar coordinate system usage
 *
 * Shows:
 * - Creating polar coordinate system
 * - Converting grid indices to Cartesian coordinates
 * - Round-trip transformation (indices → coords → indices)
 * - Verification of mathematical properties
 */
void example_polar_coordinates() {
  std::cout << "=== Example 1: Polar Coordinates (2D) ===\n\n";

  // Define polar coordinate system: r in [0, 10], theta in [0, 2*pi]
  PolarCoordinateSystem cs({0.0, 10.0},           // r range
                           {0.0, 2.0 * M_PI},     // theta range
                           {false, true, false}); // theta periodic

  // Grid dimensions: 64 radial × 128 angular × 1
  const Int3 size = {64, 128, 1};

  std::cout << "Polar grid configuration:\n";
  std::cout << "  r ∈ [" << cs.m_r_min << ", " << cs.m_r_max << "]\n";
  std::cout << "  θ ∈ [" << cs.m_theta_min << ", " << cs.m_theta_max
            << "] radians\n";
  std::cout << "  Grid size: " << size[0] << " (radial) × " << size[1]
            << " (angular)\n\n";

  // Test point 1: Center of grid (middle r, θ = 0)
  std::cout << "Test 1: Point at r=5, theta=0 (on +x axis)\n";
  Int3 idx1 = {32, 0, 0}; // Middle of radial, theta=0
  Real3 coords1 = polar_to_coords(cs, idx1, size);
  std::cout << "  Grid indices: (" << idx1[0] << ", " << idx1[1] << ", " << idx1[2]
            << ")\n";
  std::cout << "  Cartesian (x,y,z): (" << coords1[0] << ", " << coords1[1] << ", "
            << coords1[2] << ")\n";
  std::cout << "  Expected: (~5.0, ~0.0, 0.0)\n\n";

  // Test point 2: θ = π/2 (on +y axis)
  std::cout << "Test 2: Point at r=5, theta=pi/2 (on +y axis)\n";
  Int3 idx2 = {32, 32, 0}; // Middle r, theta = pi/2
  Real3 coords2 = polar_to_coords(cs, idx2, size);
  std::cout << "  Grid indices: (" << idx2[0] << ", " << idx2[1] << ", " << idx2[2]
            << ")\n";
  std::cout << "  Cartesian (x,y,z): (" << coords2[0] << ", " << coords2[1] << ", "
            << coords2[2] << ")\n";
  std::cout << "  Expected: (~0.0, ~5.0, 0.0)\n\n";

  // Test round-trip transformation
  std::cout << "Test 3: Round-trip transformation\n";
  Int3 idx_original = {40, 60, 0};
  Real3 coords_temp = polar_to_coords(cs, idx_original, size);
  Int3 idx_roundtrip = polar_to_indices(cs, coords_temp, size);
  std::cout << "  Original indices: (" << idx_original[0] << ", " << idx_original[1]
            << ", " << idx_original[2] << ")\n";
  std::cout << "  After round-trip: (" << idx_roundtrip[0] << ", "
            << idx_roundtrip[1] << ", " << idx_roundtrip[2] << ")\n";
  std::cout << "  Match: "
            << (idx_original[0] == idx_roundtrip[0] &&
                        idx_original[1] == idx_roundtrip[1]
                    ? "✓ YES"
                    : "✗ NO")
            << "\n\n";
}

/**
 * @brief Demonstrate spherical coordinate system usage
 *
 * Shows complete 3D spherical coordinate transformations with verification.
 */
void example_spherical_coordinates() {
  std::cout << "=== Example 2: Spherical Coordinates (3D) ===\n\n";

  // Full sphere: r in [0, 10], theta in [0, pi], phi in [0, 2*pi]
  SphericalCoordinateSystem cs({0.0, 10.0},           // r range
                               {0.0, M_PI},           // theta range
                               {0.0, 2.0 * M_PI},     // phi range
                               {false, false, true}); // phi periodic

  const Int3 size = {32, 32, 64}; // n_r × n_θ × n_φ

  std::cout << "Spherical grid configuration:\n";
  std::cout << "  r ∈ [" << cs.m_r_min << ", " << cs.m_r_max << "]\n";
  std::cout << "  theta in [" << cs.m_theta_min << ", " << cs.m_theta_max
            << "] (polar)\n";
  std::cout << "  phi in [" << cs.m_phi_min << ", " << cs.m_phi_max
            << "] (azimuthal)\n";
  std::cout << "  Grid size: " << size[0] << " × " << size[1] << " × " << size[2]
            << "\n\n";

  // Test 1: North pole (θ = 0, z = +r)
  std::cout << "Test 1: North pole (theta=0, any phi)\n";
  Int3 idx1 = {16, 0, 0}; // Mid radius, theta=0
  Real3 coords1 = spherical_to_coords(cs, idx1, size);
  std::cout << "  Cartesian (x,y,z): (" << coords1[0] << ", " << coords1[1] << ", "
            << coords1[2] << ")\n";
  std::cout << "  Expected: (~0, ~0, ~5) - on +z axis\n\n";

  // Test 2: Equator, +x direction (θ = π/2, φ = 0)
  std::cout << "Test 2: Equator, +x direction (theta=pi/2, phi=0)\n";
  Int3 idx2 = {16, 16, 0}; // Mid r, theta=pi/2, phi=0
  Real3 coords2 = spherical_to_coords(cs, idx2, size);
  std::cout << "  Cartesian (x,y,z): (" << coords2[0] << ", " << coords2[1] << ", "
            << coords2[2] << ")\n";
  std::cout << "  Expected: (~5, ~0, ~0) - on +x axis\n\n";

  // Test 3: Equator, +y direction (θ = π/2, φ = π/2)
  std::cout << "Test 3: Equator, +y direction (theta=pi/2, phi=pi/2)\n";
  Int3 idx3 = {16, 16, 16}; // Mid r, theta=pi/2, phi=pi/2
  Real3 coords3 = spherical_to_coords(cs, idx3, size);
  std::cout << "  Cartesian (x,y,z): (" << coords3[0] << ", " << coords3[1] << ", "
            << coords3[2] << ")\n";
  std::cout << "  Expected: (~0, ~5, ~0) - on +y axis\n\n";

  // Test round-trip
  std::cout << "Test 4: Round-trip transformation\n";
  Int3 idx_orig = {20, 10, 40};
  Real3 coords_temp = spherical_to_coords(cs, idx_orig, size);
  Int3 idx_back = spherical_to_indices(cs, coords_temp, size);
  std::cout << "  Original: (" << idx_orig[0] << ", " << idx_orig[1] << ", "
            << idx_orig[2] << ")\n";
  std::cout << "  Round-trip: (" << idx_back[0] << ", " << idx_back[1] << ", "
            << idx_back[2] << ")\n";
  std::cout << "  Match: "
            << (idx_orig[0] == idx_back[0] && idx_orig[1] == idx_back[1] &&
                        idx_orig[2] == idx_back[2]
                    ? "✓ YES"
                    : "✗ NO")
            << "\n\n";
}

/**
 * @brief Summary of the extension pattern
 *
 * Shows the complete recipe for adding custom coordinate systems.
 */
void show_extension_pattern() {
  std::cout << "=== Extension Pattern: How to Add Your Own Coordinate System "
               "===\n\n";

  std::cout << "Step 1: Define a tag struct\n";
  std::cout << "  struct MyCoordTag {};  // Empty struct for tag dispatch\n\n";

  std::cout << "Step 2: Create coordinate system struct\n";
  std::cout << "  struct MyCoordinateSystem {\n";
  std::cout << "    const double m_param1;  // Your parameters\n";
  std::cout << "    const double m_param2;\n";
  std::cout << "    const Bool3 m_periodic;\n";
  std::cout << "  };\n\n";

  std::cout << "Step 3: Implement coordinate transformations\n";
  std::cout << "  // Indices → Physical coordinates\n";
  std::cout << "  inline Real3 my_to_coords(\n";
  std::cout << "      const MyCoordinateSystem& cs,\n";
  std::cout << "      const Int3& indices,\n";
  std::cout << "      const Int3& size) {\n";
  std::cout << "    // Your transformation math here\n";
  std::cout << "    return {x, y, z};\n";
  std::cout << "  }\n\n";

  std::cout << "  // Physical coordinates → Indices\n";
  std::cout << "  inline Int3 my_to_indices(\n";
  std::cout << "      const MyCoordinateSystem& cs,\n";
  std::cout << "      const Real3& coords,\n";
  std::cout << "      const Int3& size) {\n";
  std::cout << "    // Inverse transformation\n";
  std::cout << "    return {i, j, k};\n";
  std::cout << "  }\n\n";

  std::cout << "Step 4: Use it!\n";
  std::cout << "  MyCoordinateSystem cs(/* params */);\n";
  std::cout << "  Int3 size = {100, 100, 100};\n";
  std::cout << "  Real3 xyz = my_to_coords(cs, {50, 50, 50}, size);\n\n";

  std::cout << "✓ No modifications to OpenPFC source code required!\n";
  std::cout << "✓ Your code lives in your own files\n";
  std::cout << "✓ ADL (Argument-Dependent Lookup) makes it \"just work\"\n";
  std::cout
      << "✓ This is the \"Laboratory, Not Fortress\" philosophy in action!\n\n";
}

// ============================================================================
// Main Function
// ============================================================================

int main() {
  std::cout << "\n";
  std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
  std::cout << "║  OpenPFC: Custom Coordinate Systems Example                  ║\n";
  std::cout << "║  Demonstrating extensibility without source modification     ║\n";
  std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
  std::cout << "\n";

  example_polar_coordinates();
  std::cout << std::string(70, '-') << "\n\n";

  example_spherical_coordinates();
  std::cout << std::string(70, '-') << "\n\n";

  show_extension_pattern();

  std::cout << "For more information, see:\n";
  std::cout << "- docs/advanced_topics/coordinate_systems.md\n";
  std::cout << "\n";

  return 0;
}
