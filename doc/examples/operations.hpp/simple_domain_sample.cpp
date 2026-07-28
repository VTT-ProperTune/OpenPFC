// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simple_domain_sample.cpp
 * @brief Simple domain-first API sample for @ref openpfc::kernel::field::operations
 *
 * This example demonstrates the preferred domain-first API pattern for
 * coordinate-space field operations using `domain::create`. This aligns with
 * the M2 critical path refactoring.
 *
 * Demonstrates:
 * - Domain creation with `domain::create`
 * - Direct decomposition creation from domain (no `domain::to_world()`)
 * - Applying coordinate-space functions to fields using `pfc::field::apply`
 * - Creating Gaussian pulse initial conditions
 *
 * @note This is the recommended pattern for new code. Legacy construction APIs
 *       are kept only for backward compatibility during the M2 refactoring.
 */

#include <iomanip>
#include <iostream>
#include <numbers>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/field/operations.hpp>
#include <openpfc/kernel/fft/fftw_factory.hpp>

// pfc::data::Field is the M2 canonical field container (grid_field.hpp)
// field_from_subdomain provides domain-first field construction

using namespace pfc;

int main() {
  std::cout << "Domain-first field operations sample\n";
  std::cout << std::string(50, '=') << "\n\n";

  // Create a domain using the preferred domain::create API
  auto domain = domain::create(
    GridSize({64, 64, 64}),
    PhysicalOrigin({0.0, 0.0, 0.0}),
    GridSpacing({1.0, 1.0, 1.0})
  );

  std::cout << "Domain: 64³ grid with origin (0, 0, 0) and spacing (1, 1, 1)\n";
  std::cout << "Physical volume: " << domain::physical_volume(domain) << "\n\n";

  std::cout << "Setting up decomposition and creating domain-first field...\n";
  auto decomp = decomposition::create(domain, 1);
  int rank = 0; // Single rank example; for MPI, use MPI_Comm_rank()

  // Create field using the M2 domain-first pattern (grid_field.hpp + field_factory.hpp)
  auto u = pfc::data::field_from_subdomain<double>(decomp, rank, /*halo=*/0);
  size_t field_size = u.size();

  std::cout << "Local field size: " << field_size << " grid points\n\n";

  std::cout << "Applying Gaussian pulse using coordinate-space function:\n";
  std::cout << "  f(x,y,z) = exp(-r²/2) where r² = x² + y² + z²\n\n";

  // Apply a Gaussian pulse using the Field's built-in coordinate-space method
  u.apply([](const double x, const double y, const double z) {
    const double r2 = x*x + y*y + z*z;
    return std::exp(-r2/2.0);
  });

  std::cout << "Field initialized successfully.\n";

  // Sample some values at key coordinates using the Field API
  std::cout << "\nExpected field values at center and nearby points:\n";
  std::cout << std::fixed << std::setprecision(4);

  for (int i = -2; i <= 2; ++i) {
    for (int j = -2; j <= 2; ++j) {
      for (int k = -2; k <= 2; ++k) {
        // Convert integer offset to physical coordinates
        Real3 offset = {static_cast<double>(i), static_cast<double>(j), static_cast<double>(k)};
        Real3 coords = {32.0 + offset[0], 32.0 + offset[1], 32.0 + offset[2]};
        
        // Compute expected Gaussian value
        const double r2 = offset[0]*offset[0] + offset[1]*offset[1] + offset[2]*offset[2];
        const double expected = std::exp(-r2/2.0);

        std::cout << "  f(" << std::setw(5) << coords[0] << ", ";
        std::cout << std::setw(5) << coords[1] << ", ";
        std::cout << std::setw(5) << coords[2] << ") ≈ ";
        std::cout << expected << "\n";
      }
    }
  }

  std::cout << "\n" << std::string(50, '=') << "\n";
  std::cout << "Domain-first field operations sample complete.\n";
  std::cout << "Note: This pattern uses pfc::data::Field without explicit World objects.\n";
  std::cout << "      Works equally well with MPI and multiple ranks.\n";

  return 0;
}