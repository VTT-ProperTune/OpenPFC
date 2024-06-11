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

class FixedBC : public FieldModifier {

private:
  double xwidth = 20.0;
  double alpha = 1.0;
  double m_rho_low, m_rho_high;

public:
  FixedBC() = default;

  FixedBC(double rho_low, double rho_high) : m_rho_low(rho_low), m_rho_high(rho_high) {}

  void set_rho_low(double rho_low) { m_rho_low = rho_low; }
  void set_rho_high(double rho_high) { m_rho_high = rho_high; }

  void apply(Model &m, double) override {
    const Decomposition &decomp = m.get_decomposition();
    Field &field = m.get_real_field(get_field_name());
    const World &w = m.get_world();
    Vec3<int> low = decomp.inbox.low;
    Vec3<int> high = decomp.inbox.high;

    double xpos = w.Lx * w.dx - xwidth;
    long int idx = 0;
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          double x = w.x0 + i * w.dx;
          if (std::abs(x - xpos) < xwidth) {
            double S = 1.0 / (1.0 + exp(-alpha * (x - xpos)));
            field[idx] = m_rho_low * S + m_rho_high * (1.0 - S);
          }
          idx += 1;
        }
      }
    }
  }
};

} // namespace pfc
