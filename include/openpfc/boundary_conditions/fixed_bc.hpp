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
    Field &field = m.get_field();
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
