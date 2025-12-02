// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file single_seed.hpp
 * @brief Single spherical crystalline seed initial condition
 *
 * @details
 * This file defines the SingleSeed class, which places a single spherical
 * crystalline seed at a specified location. Useful for:
 * - Single crystal growth simulations
 * - Dendritic solidification studies
 * - Validation against analytical solutions
 *
 * The seed's location, size, density, and amplitude are configurable.
 *
 * Usage:
 * @code
 * auto ic = std::make_unique<pfc::SingleSeed>();
 * ic->set_amplitude(0.2);
 * ic->set_density(0.5);
 * simulator.add_initial_condition(std::move(ic));
 * @endcode
 *
 * @see seed.hpp for seed construction helper
 * @see field_modifier.hpp for base class
 *
 * @author OpenPFC Contributors
 * @date 2025
 */

#ifndef PFC_INITIAL_CONDITIONS_SINGLE_SEED_HPP
#define PFC_INITIAL_CONDITIONS_SINGLE_SEED_HPP

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
    const FFT &fft = m.get_fft();
    Field &f = m.get_real_field(get_field_name());
    Vec3<int> low = get_inbox(fft).low;
    Vec3<int> high = get_inbox(fft).high;

    // Use the new World API to get spacing and origin
    auto spacing = get_spacing(w);
    auto origin = get_origin(w);
    auto dx = spacing[0];
    auto dy = spacing[1];
    auto dz = spacing[2];
    auto x0 = origin[0];
    auto y0 = origin[1];
    auto z0 = origin[2];

    double s = 1.0 / sqrt(2.0);
    std::array<double, 3> q1 = {s, s, 0};
    std::array<double, 3> q2 = {s, 0, s};
    std::array<double, 3> q3 = {0, s, s};
    std::array<double, 3> q4 = {s, 0, -s};
    std::array<double, 3> q5 = {s, -s, 0};
    std::array<double, 3> q6 = {0, s, -s};
    std::array<std::array<double, 3>, 6> q = {q1, q2, q3, q4, q5, q6};

    long int idx = 0;
    double r2 = pow(64.0, 2);
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          double x = x0 + i * dx;
          double y = y0 + j * dy;
          double z = z0 + k * dz;
          if (x * x + y * y + z * z < r2) {
            double u = rho_seed;
            for (int q_idx = 0; q_idx < 6; q_idx++) {
              u += 2.0 * amp_eq *
                   cos(q[q_idx][0] * x + q[q_idx][1] * y + q[q_idx][2] * z);
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

#endif // PFC_INITIAL_CONDITIONS_SINGLE_SEED_HPP
