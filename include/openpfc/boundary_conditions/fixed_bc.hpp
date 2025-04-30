// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef PFC_BOUNDARY_CONDITIONS_FIXED_BC_HPP
#define PFC_BOUNDARY_CONDITIONS_FIXED_BC_HPP

#include "../field_modifier.hpp"

namespace pfc {

class FixedBC : public FieldModifier {

private:
  double xwidth = 20.0;
  double alpha = 1.0;
  double m_rho_low, m_rho_high;
  std::string m_name = "FixedBC";

public:
  FixedBC() = default;

  FixedBC(double rho_low, double rho_high)
      : m_rho_low(rho_low), m_rho_high(rho_high) {}

  void set_rho_low(double rho_low) { m_rho_low = rho_low; }
  void set_rho_high(double rho_high) { m_rho_high = rho_high; }

  const std::string &get_modifier_name() const override { return m_name; }

  void apply(Model &m, double) override {
    const Decomposition &decomp = m.get_decomposition();
    Field &field = m.get_real_field(get_field_name());
    const World &w = m.get_world();
    Vec3<int> low = decomp.get_inbox().low;
    Vec3<int> high = decomp.get_inbox().high;

    double Lx = get_size(w, 0);
    double dx = get_spacing(w, 0);
    double x0 = get_origin(w, 0);

    double xpos = Lx * dx - xwidth;
    long int idx = 0;
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          double x = x0 + i * dx;
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

#endif // PFC_BOUNDARY_CONDITIONS_FIXED_BC_HPP
