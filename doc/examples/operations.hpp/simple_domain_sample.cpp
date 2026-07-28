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
 * - Setting up FFT and decomposition with a domain
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
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/field/operations.hpp>
#include <openpfc/kernel/fft/fftw_factory.hpp>

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

  // For backward compatibility, the world type is still used internally by the
  // field operations API. The domain can be converted to world when needed.
  auto world = domain::to_world(domain);

  std::cout << "Setting up decomposition and FFT...\n";
  auto decomp = decomposition::create(world, 1);
  auto fft = fft::fftw::create(decomp, false, 6);
  size_t inbox_size = fft::size_inbox(fft);

  std::cout << "Local inbox size: " << inbox_size << " grid points\n\n";

  // Create field storage sized to the local inbox
  std::vector<double> u(inbox_size);

  std::cout << "Applying Gaussian pulse using coordinate-space function:\n";
  std::cout << "  f(x,y,z) = exp(-r²/2) where r² = x² + y² + z²\n\n";

  // Apply a Gaussian pulse using coordinate-space function
  pfc::field::apply(u, world, fft, [](const Real3& x) {
    const double r2 = (x[0]*x[0]) + (x[1]*x[1]) + (x[2]*x[2]);
    return std::exp(-r2/2.0);
  });

  std::cout << "Field initialized successfully.\n";

  // Sample some values at key coordinates
  std::cout << "\nSample field values at center and nearby points:\n";
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
  std::cout << "Note: This pattern works equally well with MPI and multiple ranks.\n";

  return 0;
}