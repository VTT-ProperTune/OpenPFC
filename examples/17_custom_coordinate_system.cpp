// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file 17_custom_coordinate_system.cpp
 * @brief Example: Defining custom coordinate systems without modifying OpenPFC
 *
 * @details
 * This example demonstrates OpenPFC's extensibility by showing how to add
 * custom coordinate systems without modifying the library source code.
 *
 * We implement several coordinate system patterns:
 * 1. **Polar coordinates** (2D: r, θ) - with free functions
 * 2. **Spherical coordinates** (3D: r, θ, φ) - with free functions
 * 3. **User-side coordinate wrapper** ( coordinate_wrapper<D> ) - modern pattern
 *
 * ## Key Techniques
 *
 * ### ADL (Argument-Dependent Lookup) - Free Function Pattern
 *
 * OpenPFC uses ADL to find coordinate transformation functions. You define
 * functions in your namespace (or pfc namespace), and ADL ensures they're
 * found automatically.
 *
 * ### User-Side Coordinate Wrapper - Modern Pattern
 *
 * The `coordinate_wrapper<D>` class template demonstrates how users can wrap
 * a `Domain` with custom coordinate transformations using modern C++ patterns.
 * This is the recommended approach for M1 and later, as it:
 * - Works with the new `Domain` API (M1.2+)
 * - Provides a template-based, dimension-agnostic interface
 * - Encapsulates coordinate transformation logic in a reusable component
 *
 * ## Philosophy: Laboratory, Not Fortress
 *
 * This example proves you can extend OpenPFC without forking or modifying its
 * source code. This is the "laboratory" philosophy in action.
 *
 * ## How to Use This Example
 *
 * 1. Read the polar coordinate example (simpler free-function pattern)
 * 2. Study the spherical coordinate example (complete free-function pattern)
 * 3. Learn the `coordinate_wrapper<D>` pattern (modern, recommended)
 * 4. Copy the pattern for your own coordinate system
 * 5. No OpenPFC source code modifications required!
 */

#include <cmath>
#include <iostream>
#include <numbers>

// OpenPFC includes
#include <openpfc/openpfc.hpp>

using namespace pfc;

// ============================================================================
// Part 1: Polar Coordinates (2D - Free Function Pattern)
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
 */
struct PolarCoordinateSystem {
  const double m_r_min;    ///< Minimum radial distance
  const double m_r_max;    ///< Maximum radial distance
  const double m_theta_min; ///< Minimum angle (radians)
  const double m_theta_max; ///< Maximum angle (radians)
  const Bool3 m_periodic;   ///< Grid periodicity: {radial, angular, unused}

  /**
   * @brief Construct polar coordinate system
   * @param r_range Radial range [r_min, r_max]
   * @param theta_range Angular range [θ_min, θ_max] in radians
   * @param periodic Periodicity (θ is typically periodic)
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
 * Applies the polar to Cartesian transformation:
 * - r = r_min + i * dr
 * - θ = θ_min + j * dtheta
 * - x = r * cos(θ)
 * - y = r * sin(θ)
 * - z = 0
 */
inline Real3 polar_to_coords(const PolarCoordinateSystem &cs, const Int3 &indices,
                             const Int3 &size) {
  const double dr = (cs.m_r_max - cs.m_r_min) / size[0];
  const double dtheta =
      (cs.m_theta_max - cs.m_theta_min) /
      (cs.m_periodic[1] ? size[1] : size[1] - 1);

  const double r = cs.m_r_min + indices[0] * dr;
  const double theta = cs.m_theta_min + indices[1] * dtheta;

  const double x = r * std::cos(theta);
  const double y = r * std::sin(theta);
  const double z = 0.0;

  return {x, y, z};
}

/**
 * @brief Convert Cartesian coordinates to grid indices (Cartesian → Polar)
 *
 * Inverse transformation of polar_to_coords().
 */
inline Int3 polar_to_indices(const PolarCoordinateSystem &cs, const Real3 &coords,
                             const Int3 &size) {
  const double x = coords[0];
  const double y = coords[1];

  const double r = std::sqrt(x * x + y * y);
  double theta = std::atan2(y, x);

  if (theta < cs.m_theta_min) {
    theta += 2.0 * std::numbers::pi;
  }

  const double dr = (cs.m_r_max - cs.m_r_min) / size[0];
  const double dtheta =
      (cs.m_theta_max - cs.m_theta_min) / (cs.m_periodic[1] ? size[1] : size[1] - 1);

  const int i_r = static_cast<int>(std::round((r - cs.m_r_min) / dr));
  const int i_theta =
      static_cast<int>(std::round((theta - cs.m_theta_min) / dtheta));

  return {i_r, i_theta, 0};
}

// ============================================================================
// Part 2: Spherical Coordinates (3D - Free Function Pattern)
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
   * @param r_range Radial range [r_min, r_max]
   * @param theta_range Polar angle range [theta_min, theta_max]
   * @param phi_range Azimuthal angle range [phi_min, phi_max]
   * @param periodic Periodicity (phi is typically periodic)
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
 * Transforms grid indices in spherical coordinates to Cartesian:
 * - x = r * sin(θ) * cos(φ)
 * - y = r * sin(θ) * sin(φ)
 * - z = r * cos(θ)
 */
inline Real3 spherical_to_coords(const SphericalCoordinateSystem &cs,
                                 const Int3 &indices, const Int3 &size) {
  const double dr = (cs.m_r_max - cs.m_r_min) / size[0];
  const double dtheta =
      (cs.m_theta_max - cs.m_theta_min) / (cs.m_periodic[1] ? size[1] : size[1] - 1);
  const double dphi =
      (cs.m_phi_max - cs.m_phi_min) / (cs.m_periodic[2] ? size[2] : size[2] - 1);

  const double r = cs.m_r_min + indices[0] * dr;
  const double theta = cs.m_theta_min + indices[1] * dtheta;
  const double phi = cs.m_phi_min + indices[2] * dphi;

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
 * Inverse of spherical_to_coords():
 * - r = √(x² + y² + z²)
 * - theta = acos(z / r)
 * - phi = atan2(y, x)
 */
inline Int3 spherical_to_indices(const SphericalCoordinateSystem &cs,
                                 const Real3 &coords, const Int3 &size) {
  const double x = coords[0];
  const double y = coords[1];
  const double z = coords[2];

  const double r = std::sqrt(x * x + y * y + z * z);
  const double theta = r > 1e-14 ? std::acos(z / r) : 0.0;
  double phi = std::atan2(y, x);

  if (phi < cs.m_phi_min) {
    phi += 2.0 * std::numbers::pi;
  }

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
// Part 3: User-Side Coordinate Wrapper - Modern Pattern (M1+)
// ============================================================================

/**
 * @brief User-side coordinate wrapper template for custom coordinate systems
 *
 * This template demonstrates the recommended pattern for wrapping OpenPFC's
 * `Domain` with custom coordinate transformations in the M1 API. Instead of
 * relying on the removed csys tag machinery, users create their own wrapper
 * classes that handle coordinate transformations.
 *
 * @tparam D Effective spatial dimension (2 for 2D, 3 for 3D)
 *
 * The wrapper provides:
 * - A callable interface via operator() for coordinate transformations
 * - Helper methods for transformations and indexing
 * - Encapsulation of custom coordinate system parameters
 *
 * This pattern is:
 * - Template-based: works with any dimension (2D or 3D)
 * - Non-intrusive: doesn't require modifying OpenPFC source code
 * - Reusable: can be composed and extended for complex coordinate systems
 * - Modern: uses standard C++ features (classes, templates, operators)
 */
template<int D>
class coordinate_wrapper {
private:
  Domain& domain; ///< Reference to the underlying OpenPFC Domain

  /**
   * @brief Scaling factors for user-to-Domain coordinate transformation
   *
   * For each axis i: domain_coord = user_coord / scaling_factors[i]
   *
   * This allows users to define non-uniform scaling between their
   * coordinate system and the Domain's Cartesian coordinates.
   *
   * For a simple scaled coordinate system:
   * - If scaling_factors = {2.0, 2.0}, then user (1.0, 1.0) maps to domain (0.5, 0.5)
   * - If scaling_factors = {10.0, 5.0}, then user (10.0, 10.0) maps to domain (1.0, 2.0)
   */
  std::array<double, D> scaling_factors;

public:
  /**
   * @brief Construct coordinate wrapper with default scaling factors
   *
   * Default scaling factors are set to 1.0 (identity transformation),
   * meaning user coordinates map directly to Domain coordinates.
   *
   * @param d Reference to the OpenPFC Domain to wrap
   *
   * @example Unit scaling (identity)
   * @code
   * Domain domain = domain::create({64, 64, 1});
   * coordinate_wrapper<2> wrapper(domain);
   * auto domain_coords = wrapper({1.0, 2.0});  // Returns {1.0, 2.0}
   * @endcode
   */
  explicit coordinate_wrapper(Domain& d) : domain(d) {
    for (int i = 0; i < D; ++i) {
      scaling_factors[i] = 1.0;  // Default: identity transformation
    }
  }

  /**
   * @brief Construct coordinate wrapper with custom scaling factors
   *
   * @param d Reference to the OpenPFC Domain to wrap
   * @param factors Scaling factors for each dimension
   *
   * @example Non-uniform scaling
   * @code
   * // User coordinates in millimeters, Domain in meters with different aspect ratio
   * coordinate_wrapper<2> wrapper(domain, {1000.0, 500.0});
   * auto domain_coords = wrapper({100.0, 50.0});  // Returns {0.1, 0.1}
   * @endcode
   */
  coordinate_wrapper(Domain& d, const std::array<double, D>& factors)
      : domain(d), scaling_factors(factors) {}

  /**
   * @brief Transform user coordinates to Domain coordinates
   *
   * This operator() provides a callable interface for coordinate transformation.
   * It applies the scaling factors to convert user coordinates to the Domain's
   * Cartesian coordinate system.
   *
   * For example, with scaling_factors = {2.0, 2.0}:
   * - user_coord = (2.0, 4.0)
   * - domain_coord = (1.0, 2.0)  [divided by scaling factors]
   *
   * @param user_coords User coordinates in the custom coordinate system
   * @return Domain coordinates in the wrapped.Domain's Cartesian system
   *
   * @note This is the primary interface of the wrapper
   * @note The transformation is linear: domain_coord = user_coord / scaling_factors
   */
  std::array<double, D> operator()(const std::array<double, D>& user_coords) {
    std::array<double, D> domain_coords = user_coords;

    for (int i = 0; i < D; ++i) {
      domain_coords[i] = user_coords[i] / scaling_factors[i];
    }

    return domain_coords;
  }

  /**
   * @brief Transform user coordinates to Domain coordinates (explicit method)
   *
   * This provides an explicit alternative to operator(), which can be clearer
   * in some contexts or when passing function pointers.
   *
   * @param user_coords User coordinates in the custom coordinate system
   * @return Domain coordinates in the wrapped Domain's Cartesian system
   */
  std::array<double, D> user_to_domain(const std::array<double, D>& user_coords) {
    return operator()(user_coords);
  }

  /**
   * @brief Get the underlying Domain reference
   *
   * @return Reference to the wrapped Domain object
   */
  Domain& get_domain() { return domain; }
  const Domain& get_domain() const { return domain; }

  /**
   * @brief Get the current scaling factors
   *
   * @return Copy of the scaling factors array
   */
  std::array<double, D> get_scaling_factors() const { return scaling_factors; }

  /**
   * @brief Set new scaling factors
   *
   * @param factors New scaling factors to apply
   */
  void set_scaling_factors(const std::array<double, D>& factors) {
    scaling_factors = factors;
  }
};

// ============================================================================
// Part 4: Usage Examples and Demonstrations
// ============================================================================

/**
 * @brief Demonstrate polar coordinate system usage
 */
void example_polar_coordinates() {
  std::cout << "=== Example 1: Polar Coordinates (2D - Free Function Pattern) ===\n\n";

  PolarCoordinateSystem cs({0.0, 10.0},                   // r range
                           {0.0, 2.0 * std::numbers::pi}, // theta range
                           {false, true, false});         // theta periodic

  const Int3 size = {64, 128, 1};

  std::cout << "Polar grid configuration:\n";
  std::cout << "  r ∈ [" << cs.m_r_min << ", " << cs.m_r_max << "]\n";
  std::cout << "  θ ∈ [" << cs.m_theta_min << ", " << cs.m_theta_max
            << "] radians\n";
  std::cout << "  Grid size: " << size[0] << " (radial) × " << size[1]
            << " (angular)\n\n";

  // Test point: r=5, theta=0 (on +x axis)
  std::cout << "Test: Point at r=5, theta=0 (on +x axis)\n";
  Int3 idx1 = {32, 0, 0};
  Real3 coords1 = polar_to_coords(cs, idx1, size);
  std::cout << "  Grid indices: (" << idx1[0] << ", " << idx1[1] << ", " << idx1[2]
            << ")\n";
  std::cout << "  Cartesian (x,y,z): (" << coords1[0] << ", " << coords1[1] << ", "
            << coords1[2] << ")\n";
  std::cout << "  Expected: (~5.0, ~0.0, 0.0)\n\n";
}

/**
 * @brief Demonstrate spherical coordinate system usage
 */
void example_spherical_coordinates() {
  std::cout << "=== Example 2: Spherical Coordinates (3D - Free Function Pattern) ===\n\n";

  SphericalCoordinateSystem cs({0.0, 10.0},                   // r range
                               {0.0, std::numbers::pi},       // theta range
                               {0.0, 2.0 * std::numbers::pi}, // phi range
                               {false, false, true});         // phi periodic

  const Int3 size = {32, 32, 64};

  std::cout << "Spherical grid configuration:\n";
  std::cout << "  r ∈ [" << cs.m_r_min << ", " << cs.m_r_max << "]\n";
  std::cout << "  Grid size: " << size[0] << " × " << size[1] << " × " << size[2]
            << "\n\n";

  // Test: North pole (θ = 0)
  std::cout << "Test: North pole (theta=0)\n";
  Int3 idx1 = {16, 0, 0};
  Real3 coords1 = spherical_to_coords(cs, idx1, size);
  std::cout << "  Cartesian (x,y,z): (" << coords1[0] << ", " << coords1[1] << ", "
            << coords1[2] << ")\n";
  std::cout << "  Expected: (~0, ~0, ~5) - on +z axis\n\n";
}

/**
 * @brief Demonstrate user-side coordinate wrapper pattern
 *
 * This example shows the recommended M1 pattern for custom coordinate systems:
 * using `coordinate_wrapper<D>` to wrap a `Domain` with custom transformations.
 */
void example_coordinate_wrapper() {
  std::cout << "=== Example 3: User-Side Coordinate Wrapper (Modern M1 Pattern) ===\n\n";

  // Create a Domain representing a 2D physical space
  // Grid: 64x64, unit spacing, origin at (0, 0), fully periodic
  Domain domain = domain::create({64, 64, 1});

  std::cout << "Domain configuration:\n";
  std::cout << "  Size: " << domain.size[0] << "×" << domain.size[1]
            << "×" << domain.size[2] << "\n";
  std::cout << "  Spacing: (" << domain.spacing[0] << ", " << domain.spacing[1]
            << ", " << domain.spacing[2] << ")\n";
  std::cout << "  Origin: (" << domain.origin[0] << ", " << domain.origin[1]
            << ", " << domain.origin[2] << ")\n\n";

  // Example 1: Identity transformation (default scaling)
  std::cout << "--- Test 1: Identity Transformation ---\n";
  {
    coordinate_wrapper<2> wrapper(domain);

    std::array<double, 2> user_coords = {5.0, 10.0};
    auto domain_coords = wrapper(user_coords);

    std::cout << "  User coordinates: (" << user_coords[0] << ", " << user_coords[1] << ")\n";
    std::cout << "  Domain coordinates: (" << domain_coords[0] << ", " << domain_coords[1] << ")\n";
    std::cout << "  Expected: (5.0, 10.0) - identity transformation\n\n";
  }

  // Example 2: Scaled coordinate system
  std::cout << "--- Test 2: Scaled Coordinate System ---\n";
  {
    // User coordinates are in millimeters, Domain is in units
    // Scaling factor: 2.0 means 2 user units = 1 domain unit
    coordinate_wrapper<2> wrapper(domain, {2.0, 2.0});

    std::array<double, 2> user_coords = {10.0, 20.0};
    auto domain_coords = wrapper(user_coords);

    std::cout << "  Scaling factors: (" << wrapper.get_scaling_factors()[0]
              << ", " << wrapper.get_scaling_factors()[1] << ")\n";
    std::cout << "  User coordinates: (" << user_coords[0] << ", " << user_coords[1] << ")\n";
    std::cout << "  Domain coordinates: (" << domain_coords[0] << ", " << domain_coords[1] << ")\n";
    std::cout << "  Expected: (5.0, 10.0) - user coords scaled by factor 2.0\n\n";
  }

  // Example 3: Non-uniform scaling
  std::cout << "--- Test 3: Non-Uniform Scaling ---\n";
  {
    // Different scaling for x and y axes
    coordinate_wrapper<2> wrapper(domain, {10.0, 5.0});

    std::array<double, 2> user_coords = {100.0, 50.0};
    auto domain_coords = wrapper(user_coords);

    std::cout << "  Scaling factors: (" << wrapper.get_scaling_factors()[0]
              << ", " << wrapper.get_scaling_factors()[1] << ")\n";
    std::cout << "  User coordinates: (" << user_coords[0] << ", " << user_coords[1] << ")\n";
    std::cout << "  Domain coordinates: (" << domain_coords[0] << ", " << domain_coords[1] << ")\n";
    std::cout << "  Expected: (10.0, 10.0) - x scaled by 10, y scaled by 5\n\n";
  }

  // Example 4: Using explicit user_to_domain method
  std::cout << "--- Test 4: Explicit user_to_domain Method ---\n";
  {
    coordinate_wrapper<2> wrapper(domain, {3.0, 3.0});

    std::array<double, 2> user_coords = {9.0, 6.0};
    auto domain_coords = wrapper.user_to_domain(user_coords);

    std::cout << "  User coordinates: (" << user_coords[0] << ", " << user_coords[1] << ")\n";
    std::cout << "  Domain coordinates: (" << domain_coords[0] << ", " << domain_coords[1] << ")\n";
    std::cout << "  Expected: (3.0, 2.0) - using explicit method\n\n";
  }

  // Example 5: 3D wrapper
  std::cout << "--- Test 5: 3D Coordinate Wrapper ---\n";
  {
    Domain domain_3d = domain::create({32, 32, 32});
    coordinate_wrapper<3> wrapper_3d(domain_3d, {2.0, 2.0, 2.0});

    std::array<double, 3> user_coords_3d = {10.0, 14.0, 18.0};
    auto domain_coords_3d = wrapper_3d(user_coords_3d);

    std::cout << "  User coordinates: (" << user_coords_3d[0] << ", " << user_coords_3d[1]
              << ", " << user_coords_3d[2] << ")\n";
    std::cout << "  Domain coordinates: (" << domain_coords_3d[0] << ", " << domain_coords_3d[1]
              << ", " << domain_coords_3d[2] << ")\n";
    std::cout << "  Expected: (5.0, 7.0, 9.0) - 3D with uniform scaling\n\n";
  }

  std::cout << "Key advantages of the coordinate_wrapper pattern:\n";
  std::cout << "  ✓ Works with the new Domain API (M1.2+)\n";
  std::cout << "  ✓ Template-based, dimension-agnostic\n";
  std::cout << "  ✓ Encapsulates coordinate transformation logic\n";
  std::cout << "  ✓ Callable interface via operator()\n";
  std::cout << "  ✓ Extendable to complex coordinate systems\n\n";
}

/**
 * @brief Summary of extension patterns
 */
void show_extension_pattern() {
  std::cout << "=== Extension Patterns: How to Add Your Own Coordinate System ===\n\n";

  std::cout << "Pattern 1: Free Functions (Traditional, ADL-based)\n";
  std::cout << "  - Define tag struct and coordinate system value type\n";
  std::cout << "  - Implement `*_to_coords()` and `*_to_indices()` free functions\n";
  std::cout << "  - Found via ADL, no namespace qualification needed\n";
  std::cout << "  - Used in polar/spherical examples above\n\n";

  std::cout << "Pattern 2: User-Side Wrapper (Recommended for M1+)\n";
  std::cout << "  - Create `coordinate_wrapper<D>` class template\n";
  std::cout << "  - Store reference to Domain and custom parameters\n";
  std::cout << "  - Implement operator() for coordinate transformations\n";
  std::cout << "  - Encapsulates all transformation logic\n";
  std::cout << "  - Works with new Domain API\n\n";

  std::cout << "Choosing a pattern:\n";
  std::cout << "  - Use free functions for simple coordinate systems (polar, spherical)\n";
  std::cout << "  - Use coordinate_wrapper<D> for complex, reusable coordinates\n";
  std::cout << "  - coordinate_wrapper<D> integrates better with M1 Domain API\n\n";

  std::cout << "✓ No modifications to OpenPFC source code required!\n";
  std::cout << "✓ Both patterns follow the \"Laboratory, Not Fortress\" philosophy\n\n";
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

  example_coordinate_wrapper();
  std::cout << std::string(70, '-') << "\n\n";

  show_extension_pattern();

  std::cout << "For more information, see:\n";
  std::cout << "- docs/advanced_topics/coordinate_systems.md\n";
  std::cout << "- include/openpfc/kernel/data/domain.hpp (M1 Domain API)\n";
  std::cout << "\n";

  return 0;
}
