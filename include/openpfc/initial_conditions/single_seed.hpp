/*

OpenPFC, a simulation software for the phase field crystal method.
Copyright (C) 2024 VTT Technical Research Centre of Finland Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.

*/

#pragma once

#include "../field_modifier.hpp"

namespace pfc {

class SingleSeed : public FieldModifier {
  double amp_eq, rho_seed;

public:
  void set_amplitude(double amplitude) { amp_eq = amplitude; }
  double get_amplitude() const { return amp_eq; }

  void set_density(double density) { rho_seed = density; }
  double get_density() const { return rho_seed; }

  void apply(Model &m, double) override {
    const World &w = m.get_world();
    const Decomposition &decomp = m.get_decomposition();
    Field &f = m.get_real_field(get_field_name());
    Vec3<int> low = decomp.inbox.low;
    Vec3<int> high = decomp.inbox.high;
    auto dx = w.dx;
    auto dy = w.dy;
    auto dz = w.dz;
    auto x0 = w.x0;
    auto y0 = w.y0;
    auto z0 = w.z0;

    double s = 1.0 / sqrt(2.0);
    std::array<double, 3> q1 = {s, s, 0};
    std::array<double, 3> q2 = {s, 0, s};
    std::array<double, 3> q3 = {0, s, s};
    std::array<double, 3> q4 = {s, 0, -s};
    std::array<double, 3> q5 = {s, -s, 0};
    std::array<double, 3> q6 = {0, s, -s};
    std::array<std::array<double, 3>, 6> q = {q1, q2, q3, q4, q5, q6};

    long int idx = 0;
    // double r2 = pow(0.2 * (Lx * dx), 2);
    double r2 = pow(64.0, 2);
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          double x = x0 + i * dx;
          double y = y0 + j * dy;
          double z = z0 + k * dz;
          if (x * x + y * y + z * z < r2) {
            double u = rho_seed;
            for (int i = 0; i < 6; i++) {
              u += 2.0 * amp_eq * cos(q[i][0] * x + q[i][1] * y + q[i][2] * z);
            }
            f[idx] = u;
          }
          idx += 1;
        }
      }
    }
  }
};

} // namespace pfc
