// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file 14_custom_field_initializer.cpp
 * @brief Example: Custom field initialization with pfc::data::Field
 *
 * @details
 * This example demonstrates how to create custom field initialization patterns
 * using pfc::data::Field and Field::apply(). We show three physical patterns:
 *
 * 1. **Lamb-Oseen Vortex** - Rotating fluid vortex with viscous core
 * 2. **Gaussian Bump** - Localized concentration or temperature field
 * 3. **Checkerboard** - Periodic alternating pattern
 *
 * ## Key Concept: Field::apply() with Callables
 *
 * pfc::data::Field::apply() accepts any callable that can be invoked with
 * physical coordinates (double x, double y, double z) or (const Real3& pos).
 * This enables flexible, expressive initialization patterns using lambdas,
 * functors, or free functions.
 *
 * ## Integration with OpenPFC
 *
 * See Field from decomposition using field_from_subdomain_unpadded factory:
 * - examples/08_discrete_fields.cpp - Basic Field usage
 * - examples/19_explicit_stepper_fd.cpp - Physics application with custom initialization
 */

#include <cmath>
#include <iostream>
#include <numbers>
#include <mpi.h>

#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>

using namespace pfc;
using pfc::data::field_from_subdomain_unpadded;
using pfc::data::field_from_subdomain;

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
  Real3 m_center;       ///< Vortex center (x, y, z)
  double m_strength;    ///< Circulation Γ
  double m_core_radius; ///< Core radius r_c

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
  Real3 m_center;     ///< Peak location
  double m_amplitude; ///< Peak height A
  double m_width;     ///< Standard deviation σ

  GaussianBump(Real3 center, double amplitude, double width)
      : m_center(center), m_amplitude(amplitude), m_width(width) {}
};

/**
 * @brief 3D checkerboard pattern
 *
 * Periodic alternating values, useful for testing and validation.
 */
struct CheckerboardPattern {
  double m_value_high; ///< Value in "white" cells
  double m_value_low;  ///< Value in "black" cells
  Real3 m_period;      ///< Period in each direction

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
double evaluate_vortex(const my_project::VortexPattern &pattern, const Real3 &pos) {
  // Distance from vortex center in x-y plane
  double dx = pos[0] - pattern.m_center[0];
  double dy = pos[1] - pattern.m_center[1];
  double r = std::sqrt(dx * dx + dy * dy);

  // Lamb-Oseen vortex profile
  double r_c_sq = pattern.m_core_radius * pattern.m_core_radius;
  double value = 0.0;

  if (r > 1e-10) { // Avoid division by zero at center
    value = (pattern.m_strength / (2.0 * std::numbers::pi * r)) *
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
double evaluate_gaussian(const my_project::GaussianBump &pattern, const Real3 &pos) {
  // Distance from center
  double dx = pos[0] - pattern.m_center[0];
  double dy = pos[1] - pattern.m_center[1];
  double dz = pos[2] - pattern.m_center[2];
  double dist_sq = dx * dx + dy * dy + dz * dz;

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
double evaluate_checkerboard(const my_project::CheckerboardPattern &pattern,
                             const Real3 &pos) {
  // Determine which cell of the checkerboard
  int cell_i = static_cast<int>(std::floor(pos[0] / pattern.m_period[0]));
  int cell_j = static_cast<int>(std::floor(pos[1] / pattern.m_period[1]));
  int cell_k = static_cast<int>(std::floor(pos[2] / pattern.m_period[2]));

  // Checkerboard: alternate based on sum of cell indices
  int sum = cell_i + cell_j + cell_k;
  return (sum % 2 == 0) ? pattern.m_value_high : pattern.m_value_low;
}

// ============================================================================
// Part 3: Usage Examples with pfc::data::Field
// ============================================================================

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  std::cout << "\n";
  std::cout << "╔════════════════════════════════════════════════════════╗\n";
  std::cout << "║  OpenPFC: Custom Field Initialization Patterns        ║\n";
  std::cout << "╚════════════════════════════════════════════════════════╝\n";
  std::cout << "\n";

  // Example 1: Vortex Pattern Initialization
  std::cout << "=== Example 1: Vortex Pattern ===\n\n";

  // Create domain and decomposition using strong types
  auto vortex_domain = domain::create(pfc::GridSize({32, 32, 1}),
                                       pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                       pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto vortex_decomp = decomposition::create(vortex_domain, nproc);

  // Create Field using factory function
  auto vortex_field = field_from_subdomain_unpadded<double>(vortex_decomp, rank);

  // Initialize with vortex pattern using lambda
  const double vortex_strength = 5.0;
  const double vortex_core_radius = 2.0;
  vortex_field.apply([vortex_strength, vortex_core_radius](double x, double y, double z) -> double {
    double r = std::sqrt(x * x + y * y);  // Distance from center in x-y plane
    double r_c_sq = vortex_core_radius * vortex_core_radius;
    double value = 0.0;

    if (r > 1e-10) {  // Avoid division by zero at center
      value = (vortex_strength / (2.0 * std::numbers::pi * r)) *
              (1.0 - std::exp(-r * r / r_c_sq));
    }
    return value;
  });

  // Sample and display vortex profile (on rank 0)
  if (rank == 0) {
    std::cout << "Vortex tangential velocity profile:\n";
    my_project::VortexPattern vortex({0.0, 0.0, 0.0}, vortex_strength, vortex_core_radius);
    for (double r = 0.0; r <= 10.0; r += 2.0) {
      Real3 pos{r, 0.0, 0.0};  // Along x-axis
      double velocity = evaluate_vortex(vortex, pos);
      std::cout << "  r = " << r << " : v_θ = " << velocity << "\n";
    }
    std::cout << "\n✅ Vortex field initialized with Field::apply()\n\n";
  }

  // Example 2: Gaussian Bump Initialization
  std::cout << "=== Example 2: Gaussian Bump ===\n\n";

  // Create domain and decomposition for Gaussian
  auto gaussian_domain = domain::create(pfc::GridSize({16, 16, 16}),
                                        pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                        pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto gaussian_decomp = decomposition::create(gaussian_domain, nproc);

  // Create Field using factory function
  auto gaussian_field = field_from_subdomain_unpadded<double>(gaussian_decomp, rank);

  // Initialize with Gaussian bump using functor
  const double gaussian_amplitude = 1.0;
  const double gaussian_width = 1.5;
  const Real3 gaussian_center{0.0, 0.0, 0.0};

  // Create a functor for the Gaussian pattern
  struct GaussianInitializer {
    Real3 center;
    double amplitude;
    double width_sq;  // Precomputed sigma^2

    double operator()(double x, double y, double z) const {
      double dx = x - center[0];
      double dy = y - center[1];
      double dz = z - center[2];
      double dist_sq = dx * dx + dy * dy + dz * dz;
      return amplitude * std::exp(-dist_sq / (2.0 * width_sq));
    }
  };

  GaussianInitializer gaussian_init{gaussian_center, gaussian_amplitude, gaussian_width * gaussian_width};
  gaussian_field.apply(gaussian_init);

  // Display Gaussian profile (on rank 0)
  if (rank == 0) {
    std::cout << "Gaussian profile:\n";
    my_project::GaussianBump bump(gaussian_center, gaussian_amplitude, gaussian_width);
    for (double x = 0.0; x <= 5.0; x += 1.0) {
      Real3 pos{x, 0.0, 0.0};
      double value = evaluate_gaussian(bump, pos);
      std::cout << "  x = " << x << " : φ = " << value << "\n";
    }
    std::cout << "\n✅ Gaussian field initialized with functor\n\n";
  }

  // Example 3: Checkerboard Pattern Initialization
  std::cout << "=== Example 3: Checkerboard Pattern ===\n\n";

  // Create domain and decomposition for checkerboard
  auto checker_domain = domain::create(pfc::GridSize({8, 8, 8}),
                                       pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                       pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto checker_decomp = decomposition::create(checker_domain, nproc);

  // Create Field using factory function
  auto checker_field = field_from_subdomain_unpadded<double>(checker_decomp, rank);

  // Initialize with checkerboard pattern using free function wrapper
  const double checker_high = 1.0;
  const double checker_low = -1.0;
  const Real3 checker_period{2.0, 2.0, 2.0};

  checker_field.apply([checker_high, checker_low, checker_period](double x, double y, double z) -> double {
    int cell_i = static_cast<int>(std::floor(x / checker_period[0]));
    int cell_j = static_cast<int>(std::floor(y / checker_period[1]));
    int cell_k = static_cast<int>(std::floor(z / checker_period[2]));

    // Checkerboard: alternate based on sum of cell indices
    int sum = cell_i + cell_j + cell_k;
    return (sum % 2 == 0) ? checker_high : checker_low;
  });

  // Display checkerboard pattern sample (on rank 0)
  if (rank == 0) {
    std::cout << "Checkerboard pattern (z=0 plane):\n";
    my_project::CheckerboardPattern checker(checker_high, checker_low, checker_period);
    for (int j = 0; j < 4; ++j) {
      std::cout << "  ";
      for (int i = 0; i < 4; ++i) {
        Real3 pos{i * 1.0, j * 1.0, 0.0};
        double value = evaluate_checkerboard(checker, pos);
        std::cout << (value > 0 ? "+" : "-") << "  ";
      }
      std::cout << "\n";
    }
    std::cout << "\n✅ Checkerboard field initialized with lambda\n\n";
  }

  if (rank == 0) {
    std::cout << "✅ All custom field initialization examples completed!\n";
    std::cout << "\n";
    std::cout << "📖 Key patterns demonstrated:\n";
    std::cout << "   • field_from_subdomain_unpadded() - Create Field from decomposition\n";
    std::cout << "   • Field::apply() - Initialize field with callable patterns\n";
    std::cout << "   • Lambda expressions - Inline pattern definitions\n";
    std::cout << "   • Functors - Reusable pattern objects with state\n";
    std::cout << "\n";
  }


  MPI_Finalize();
  return 0;
}
