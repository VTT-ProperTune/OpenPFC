// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulation_factories.hpp
 * @brief Domain-based fixtures for simulation model tests (M1 migration)
 */

#ifndef OPENPFC_TESTS_FIXTURES_SIMULATION_FACTORIES_HPP
#define OPENPFC_TESTS_FIXTURES_SIMULATION_FACTORIES_HPP

#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>

namespace pfc::test {

// =============================================================================
// DomainFactory - Factory for creating Domain objects with consistent parameters
// =============================================================================

class DomainFactory {
public:
  /**
   * @brief Create a default domain for basic simulation tests
   * @param nx Grid size in x direction
   * @param ny Grid size in y direction
   * @param nz Grid size in z direction
   * @return Domain with unit spacing, zero origin, fully periodic
   */
  static Domain create_default_domain(int nx = 32, int ny = 32, int nz = 32) {
    return domain::create({nx, ny, nz});
  }

  /**
   * @brief Create FFT-compatible domain (sizes divisible by 2 for HeFFTe)
   * @param nx Grid size in x direction (must be even for HeFFTe)
   * @param ny Grid size in y direction (must be even for HeFFTe)
   * @param nz Grid size in z direction (must be even for HeFFTe)
   * @return Domain with unit spacing, zero origin, fully periodic
   */
  static Domain create_fft_domain(int nx = 32, int ny = 32, int nz = 32) {
    // Ensure sizes are divisible by 2 for HeFFTe
    return create_default_domain(nx, ny, nz);
  }

  /**
   * @brief Create domain with custom grid size, origin, and spacing
   * @param size Grid size
   * @param origin Physical origin
   * @param spacing Grid spacing
   * @param periodic Per-axis periodicity flags
   * @return Domain with specified parameters
   */
  static Domain create_custom_domain(
      const pfc::types::Int3 &size,
      const pfc::types::Real3 &origin,
      const pfc::types::Real3 &spacing,
      const pfc::types::Bool3 &periodic = {true, true, true}) {
    return domain::create(GridSize(size), PhysicalOrigin(origin), GridSpacing(spacing), periodic);
  }

  /**
   * @brief Create domain with custom grid size and spacing (origin at zero)
   * @param size Grid size
   * @param spacing Grid spacing
   * @param periodic Per-axis periodicity flags
   * @return Domain with specified parameters
   */
  static Domain create_domain_with_spacing(
      const pfc::types::Int3 &size,
      const pfc::types::Real3 &spacing,
      const pfc::types::Bool3 &periodic = {true, true, true}) {
    return domain::create(GridSize(size), PhysicalOrigin({0.0, 0.0, 0.0}), GridSpacing(spacing), periodic);
  }
};

// =============================================================================
// SimulationModelFixture - Test fixture providing Domain access
// =============================================================================

/**
 * @brief Test fixture providing Domain access (compatible with Catch2)
 *
 * This fixture is designed to be used as a base class for Catch2 test fixtures.
 * It provides a configured Domain object and setup methods for different
 * domain configurations.
 */
class SimulationModelFixture {
public:
  /** @brief Access to the domain object */
  Domain& domain() { return domain_; }
  const Domain& domain() const { return domain_; }

protected:
  Domain domain_;

  /** @brief Set up default domain (32³ grid, unit spacing, zero origin) */
  void SetUpDefaultDomain(int nx = 32, int ny = 32, int nz = 32) {
    domain_ = DomainFactory::create_default_domain(nx, ny, nz);
  }

  /** @brief Set up FFT-compatible domain (sizes divisible by 2) */
  void SetUpFFTDomain(int nx = 32, int ny = 32, int nz = 32) {
    domain_ = DomainFactory::create_fft_domain(nx, ny, nz);
  }

  /** @brief Set up custom domain with size, origin, and spacing */
  void SetUpCustomDomain(
      const pfc::types::Int3 &size,
      const pfc::types::Real3 &origin,
      const pfc::types::Real3 &spacing,
      const pfc::types::Bool3 &periodic = {true, true, true}) {
    domain_ = DomainFactory::create_custom_domain(size, origin, spacing, periodic);
  }

  /** @brief Set up domain with size and spacing (origin at zero) */
  void SetUpCustomDomainWithSpacing(
      const pfc::types::Int3 &size,
      const pfc::types::Real3 &spacing,
      const pfc::types::Bool3 &periodic = {true, true, true}) {
    domain_ = DomainFactory::create_domain_with_spacing(size, spacing, periodic);
  }
};

// =============================================================================
// Helper functions
// =============================================================================

/** @brief Single-domain decomposition (one MPI rank owns all) */
[[nodiscard]] inline pfc::decomposition::Decomposition
make_serial_decomposition(const Domain &domain) {
  return pfc::decomposition::create(domain, 1);
}

} // namespace pfc::test

#endif // OPENPFC_TESTS_FIXTURES_SIMULATION_FACTORIES_HPP
