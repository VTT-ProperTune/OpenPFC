// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file 14_custom_field_initializer.cpp
 * @brief Example: Custom field initialization patterns
 *
 * @details
 * This example demonstrates how to create custom field initialization patterns
 * using simple structs and evaluation functions. We show three physical patterns:
 * 
 * 1. **Lamb-Oseen Vortex** - Rotating fluid vortex with viscous core
 * 2. **Gaussian Bump** - Localized concentration or temperature field
 * 3. **Checkerboard** - Periodic alternating pattern
 *
 * ## Key Concept: Struct-Based Patterns
 *
 * Instead of inheritance hierarchies, we use simple structs with public data.
 * This follows OpenPFC's "structs + free functions" philosophy.
 *
 * ## Integration with OpenPFC
 *
 * For full integration with DiscreteField/World APIs using ADL, see:
 * - docs/extending_openpfc/adl_extension_patterns.md
 * - examples/17_custom_coordinate_system.cpp
 *
 * This simplified example focuses on the pattern concept itself.
 */

#include <openpfc/openpfc.hpp>
#include <cmath>
#include <iostream>

using namespace pfc;

// Use M_PI constant
#ifndef M_PI
constexpr double M_PI = 3.14159265358979323846;
#endif

// ============================================================================
// Part 1: Define Custom Pattern Types
// ============================================================================

/**
 * Custom namespace for user-defined patterns.
 */
namespace my_project {

/**
 * @brief Lamb-Oseen vortex pattern
 * 
 * Models a rotating vortex with viscous core, common in fluid dynamics.
 * The tangential velocity follows: v_θ(r) = (Γ/2πr)[1 - exp(-r²/r_c²)]
 * 
 * @see https://en.wikipedia.org/wiki/Lamb%E2%80%93Oseen_vortex
 */
struct VortexPattern {
    Real3 m_center;           ///< Vortex center (x, y, z)
    double m_strength;        ///< Circulation Γ
    double m_core_radius;     ///< Core radius r_c
    
    VortexPattern(Real3 center, double strength, double core_radius)
        : m_center(center), m_strength(strength), m_core_radius(core_radius) {}
};

/**
 * @brief 3D Gaussian bump
 * 
 * Models localized concentration, temperature, or density field.
 * φ(r) = A * exp(-r²/(2σ²))
 */
struct GaussianBump {
    Real3 m_center;           ///< Peak location
    double m_amplitude;       ///< Peak height A
    double m_width;           ///< Standard deviation σ
    
    GaussianBump(Real3 center, double amplitude, double width)
        : m_center(center), m_amplitude(amplitude), m_width(width) {}
};

/**
 * @brief 3D checkerboard pattern
 * 
 * Periodic alternating values, useful for testing and validation.
 */
struct CheckerboardPattern {
    double m_value_high;      ///< Value in "white" cells
    double m_value_low;       ///< Value in "black" cells
    Real3 m_period;           ///< Period in each direction
    
    CheckerboardPattern(double high, double low, Real3 period)
        : m_value_high(high), m_value_low(low), m_period(period) {}
};

} // namespace my_project

// ============================================================================
// Part 2: Evaluation Functions
// ============================================================================

/**
 * @brief Evaluate vortex pattern at given position
 * @param pattern The vortex configuration
 * @param pos Physical position to evaluate at
 * @return Tangential velocity at position
 */
double evaluate_vortex(const my_project::VortexPattern& pattern, const Real3& pos) {
    // Distance from vortex center in x-y plane
    double dx = pos[0] - pattern.m_center[0];
    double dy = pos[1] - pattern.m_center[1];
    double r = std::sqrt(dx * dx + dy * dy);
    
    // Lamb-Oseen vortex profile
    double r_c_sq = pattern.m_core_radius * pattern.m_core_radius;
    double value = 0.0;
    
    if (r > 1e-10) {  // Avoid division by zero at center
        value = (pattern.m_strength / (2.0 * M_PI * r)) *
                (1.0 - std::exp(-r * r / r_c_sq));
    }
    
    return value;
}

/**
 * @brief Evaluate Gaussian bump at given position
 * @param pattern The Gaussian configuration
 * @param pos Physical position to evaluate at
 * @return Field value at position
 */
double evaluate_gaussian(const my_project::GaussianBump& pattern, const Real3& pos) {
    // Distance from center
    double dx = pos[0] - pattern.m_center[0];
    double dy = pos[1] - pattern.m_center[1];
    double dz = pos[2] - pattern.m_center[2];
    double dist_sq = dx*dx + dy*dy + dz*dz;
    
    // Gaussian: φ = A * exp(-dist² / (2σ²))
    double sigma_sq = pattern.m_width * pattern.m_width;
    return pattern.m_amplitude * std::exp(-dist_sq / (2.0 * sigma_sq));
}

/**
 * @brief Evaluate checkerboard at given position
 * @param pattern The checkerboard configuration
 * @param pos Physical position to evaluate at
 * @return High or low value depending on position
 */
double evaluate_checkerboard(const my_project::CheckerboardPattern& pattern, const Real3& pos) {
    // Determine which cell of the checkerboard
    int cell_i = static_cast<int>(std::floor(pos[0] / pattern.m_period[0]));
    int cell_j = static_cast<int>(std::floor(pos[1] / pattern.m_period[1]));
    int cell_k = static_cast<int>(std::floor(pos[2] / pattern.m_period[2]));
    
    // Checkerboard: alternate based on sum of cell indices
    int sum = cell_i + cell_j + cell_k;
    return (sum % 2 == 0) ? pattern.m_value_high : pattern.m_value_low;
}

// ============================================================================
// Part 3: Usage Examples
// ============================================================================

void example_vortex_pattern() {
    std::cout << "=== Example 1: Vortex Pattern ===\n\n";
    
    // Define vortex at origin with strength 5.0 and core radius 2.0
    my_project::VortexPattern vortex({0.0, 0.0, 0.0}, 5.0, 2.0);
    
    // Evaluate at several radial distances
    std::cout << "Vortex tangential velocity profile:\n";
    for (double r = 0.0; r <= 10.0; r += 2.0) {
        Real3 pos{r, 0.0, 0.0};  // Along x-axis
        double velocity = evaluate_vortex(vortex, pos);
        std::cout << "  r = " << r << " : v_θ = " << velocity << "\n";
    }
    std::cout << "\n";
}

void example_gaussian_bump() {
    std::cout << "=== Example 2: Gaussian Bump ===\n\n";
    
    // Create Gaussian peak at origin with amplitude 1.0, width 1.5
    my_project::GaussianBump bump({0.0, 0.0, 0.0}, 1.0, 1.5);
    
    // Evaluate along a line
    std::cout << "Gaussian profile:\n";
    for (double x = 0.0; x <= 5.0; x += 1.0) {
        Real3 pos{x, 0.0, 0.0};
        double value = evaluate_gaussian(bump, pos);
        std::cout << "  x = " << x << " : φ = " << value << "\n";
    }
    std::cout << "\n";
}

void example_checkerboard() {
    std::cout << "=== Example 3: Checkerboard Pattern ===\n\n";
    
    // Create checkerboard with period 2.0 in each direction
    my_project::CheckerboardPattern checker(1.0, -1.0, {2.0, 2.0, 2.0});
    
    // Sample a 2D slice (z=0 plane)
    std::cout << "Checkerboard pattern (z=0 plane):\n";
    for (int j = 0; j < 4; ++j) {
        std::cout << "  ";
        for (int i = 0; i < 4; ++i) {
            Real3 pos{i * 1.0, j * 1.0, 0.0};
            double value = evaluate_checkerboard(checker, pos);
            std::cout << (value > 0 ? "+" : "-") << "  ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  OpenPFC: Custom Field Initialization Patterns Example ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    
    example_vortex_pattern();
    example_gaussian_bump();
    example_checkerboard();
    
    std::cout << "✅ All examples completed successfully!\n";
    std::cout << "\n";
    std::cout << "📖 For full integration with DiscreteField/World using ADL, see:\n";
    std::cout << "   - docs/extending_openpfc/adl_extension_patterns.md\n";
    std::cout << "   - examples/17_custom_coordinate_system.cpp\n";
    std::cout << "\n";
    
    return 0;
}
