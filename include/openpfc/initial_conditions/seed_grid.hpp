#pragma once

#include <random>

#include "../field_modifier.hpp"
#include "seed.hpp"

namespace pfc {

class SeedGrid : public FieldModifier {
private:
  int m_Nx, m_Ny, m_Nz;
  double m_X0, m_radius;

public:
  double rho, amplitude;

  SeedGrid(int Ny, int Nz, double X0, double radius)
      : m_Nx(1), m_Ny(Ny), m_Nz(Nz), m_X0(X0), m_radius(radius) {}

  void apply(Model &m, double) override {
    const World &w = m.get_world();
    const Decomposition &decomp = m.get_decomposition();
    Field &field = m.get_field();
    Vec3<int> low = decomp.inbox.low;
    Vec3<int> high = decomp.inbox.high;

    // auto Lx = w.Lx;
    auto Ly = w.Ly;
    auto Lz = w.Lz;
    auto dx = w.dx;
    auto dy = w.dy;
    auto dz = w.dz;
    auto x0 = w.x0;
    auto y0 = w.y0;
    auto z0 = w.z0;

    std::vector<Seed> seeds;

    int Nx = m_Nx;
    int Ny = m_Ny;
    int Nz = m_Nz;
    double radius = m_radius;

    double Dy = dy * Ly / Ny;
    double Dz = dz * Lz / Nz;
    double X0 = m_X0;
    double Y0 = Dy / 2.0;
    double Z0 = Dz / 2.0;
    int nseeds = Nx * Ny * Nz;

    std::cout << "Generating " << nseeds << " regular seeds with radius "
              << radius << "\n";

    srand(42);
    std::uniform_real_distribution<double> rt(-0.2 * radius, 0.2 * radius);
    std::uniform_real_distribution<double> rr(0.0, 8.0 * atan(1.0));
    std::default_random_engine re;

    for (int j = 0; j < Ny; j++) {
      for (int k = 0; k < Nz; k++) {
        const std::array<double, 3> location = {
            X0 + rt(re), Y0 + Dy * j + rt(re), Z0 + Dz * k + rt(re)};
        const std::array<double, 3> orientation = {rr(re), rr(re), rr(re)};
        const Seed seed(location, orientation, radius, rho, amplitude);
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
