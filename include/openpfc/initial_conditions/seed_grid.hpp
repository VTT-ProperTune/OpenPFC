// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file seed_grid.hpp
 * @brief Regular grid of crystalline seeds initial condition
 *
 * @details
 * This file defines the SeedGrid class, which places spherical crystalline seeds
 * in a regular 3D grid pattern. Useful for:
 * - Studying grain growth with controlled initial microstructure
 * - Investigating grain boundary interactions
 * - Polycrystalline solidification simulations
 *
 * The grid spacing and seed properties (size, density, amplitude) are configurable.
 *
 * Usage:
 * @code
 * auto ic = std::make_unique<pfc::SeedGrid>();
 * ic->set_Nx(4);  // 4x4x4 grid
 * ic->set_Ny(4);
 * ic->set_Nz(4);
 * ic->set_radius(5.0);
 * simulator.add_initial_condition(std::move(ic));
 * @endcode
 *
 * @see seed.hpp for underlying seed construction
 * @see field_modifier.hpp for base class
 *
 * @author OpenPFC Contributors
 * @date 2025
 */

#ifndef PFC_INITIAL_CONDITIONS_SEED_GRID_HPP
#define PFC_INITIAL_CONDITIONS_SEED_GRID_HPP

#include <random>

#include "../field_modifier.hpp"
#include "openpfc/field/operations.hpp"
#include "seed.hpp"

namespace pfc {

class SeedGrid : public FieldModifier {
private:
  int m_Nx = 1, m_Ny = 2, m_Nz = 2;
  double m_X0, m_radius;
  double m_rho, m_amplitude;

public:
  // Setters
  void set_Nx(int Nx) { m_Nx = Nx; }
  void set_Ny(int Ny) { m_Ny = Ny; }
  void set_Nz(int Nz) { m_Nz = Nz; }
  void set_X0(double X0) { m_X0 = X0; }
  void set_radius(double radius) { m_radius = radius; }
  void set_density(double rho) { m_rho = rho; }
  void set_amplitude(double amplitude) { m_amplitude = amplitude; }

  // Getters
  int get_Nx() const { return m_Nx; }
  int get_Ny() const { return m_Ny; }
  int get_Nz() const { return m_Nz; }
  double get_X0() const { return m_X0; }
  double get_radius() const { return m_radius; }
  double get_density() const { return m_rho; }
  double get_amplitude() const { return m_amplitude; }

  SeedGrid() = default;

  SeedGrid(int Ny, int Nz, double X0, double radius)
      : m_Nx(1), m_Ny(Ny), m_Nz(Nz), m_X0(X0), m_radius(radius) {}

  void apply(Model &m, double) override {
    // Functional coordinate-space implementation using field::apply
    const World &w = m.get_world();
    const auto size = get_size(w);
    const auto spacing = get_spacing(w);

    std::vector<Seed> seeds;
    const int Nx = m_Nx;
    const int Ny = m_Ny;
    const int Nz = m_Nz;
    const double radius = get_radius();

    const double Dy = spacing[1] * size[1] / Ny;
    const double Dz = spacing[2] * size[2] / Nz;
    const double X0 = m_X0;
    const double Y0 = Dy / 2.0;
    const double Z0 = Dz / 2.0;
    const int nseeds = Nx * Ny * Nz;

    std::cout << "Generating " << nseeds << " regular seeds with radius " << radius
              << "\n";

    std::mt19937_64 re(42);
    std::uniform_real_distribution<double> rt(-0.2 * radius, 0.2 * radius);
    std::uniform_real_distribution<double> rr(0.0, 8.0 * atan(1.0));

    for (int j = 0; j < Ny; j++) {
      for (int k = 0; k < Nz; k++) {
        const std::array<double, 3> location = {X0 + rt(re), Y0 + Dy * j + rt(re),
                                                Z0 + Dz * k + rt(re)};
        const std::array<double, 3> orientation = {rr(re), rr(re), rr(re)};
        const Seed seed(location, orientation, get_radius(), get_density(),
                        get_amplitude());
        seeds.push_back(seed);
      }
    }

    pfc::field::apply(m, get_field_name(), [seeds](const pfc::Real3 &X) {
      for (const auto &seed : seeds) {
        if (seed.is_inside(X)) {
          return seed.get_value(X);
        }
      }
      return 0.0; // Outside all seeds
    });
  }
};

} // namespace pfc

#endif // PFC_INITIAL_CONDITIONS_SEED_GRID_HPP
