// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef PFC_INITIAL_CONDITIONS_SEED_GRID_HPP
#define PFC_INITIAL_CONDITIONS_SEED_GRID_HPP

#include <random>

#include "../field_modifier.hpp"
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
    const World &w = m.get_world();
    const Decomposition &decomp = m.get_decomposition();
    Field &field = m.get_real_field(get_field_name());
    Int3 low = get_inbox(decomp).low;
    Int3 high = get_inbox(decomp).high;

    // Use the new World API to get size, spacing, and origin
    auto size = get_size(w);
    auto spacing = get_spacing(w);
    auto origin = get_origin(w);
    auto Ly = size[1];
    auto Lz = size[2];
    auto dx = spacing[0];
    auto dy = spacing[1];
    auto dz = spacing[2];
    auto x0 = origin[0];
    auto y0 = origin[1];
    auto z0 = origin[2];

    std::vector<Seed> seeds;

    int Nx = m_Nx;
    int Ny = m_Ny;
    int Nz = m_Nz;
    double radius = get_radius();

    double Dy = dy * Ly / Ny;
    double Dz = dz * Lz / Nz;
    double X0 = m_X0;
    double Y0 = Dy / 2.0;
    double Z0 = Dz / 2.0;
    int nseeds = Nx * Ny * Nz;

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

    long int idx = 0;
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          const double x = x0 + i * dx;
          const double y = y0 + j * dy;
          const double z = z0 + k * dz;
          const std::array<double, 3> X = {x, y, z};
          for (const auto &seed : seeds) {
            if (seed.is_inside(X)) {
              field[idx] = seed.get_value(X);
              break;
            }
          }
          idx += 1;
        }
      }
    }
  }
};

} // namespace pfc

#endif // PFC_INITIAL_CONDITIONS_SEED_GRID_HPP
