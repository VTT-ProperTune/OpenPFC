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
#include "openpfc/field/operations.hpp"

namespace pfc {

class SingleSeed : public FieldModifier {
  double amp_eq, rho_seed;

public:
  void set_amplitude(double amplitude) { amp_eq = amplitude; }
  double get_amplitude() const { return amp_eq; }

  void set_density(double density) { rho_seed = density; }
  double get_density() const { return rho_seed; }

  void apply(Model &m, double) override {
    // Functional coordinate-space implementation using field::apply
    const double s = 1.0 / sqrt(2.0);
    const std::array<double, 3> q1 = {s, s, 0};
    const std::array<double, 3> q2 = {s, 0, s};
    const std::array<double, 3> q3 = {0, s, s};
    const std::array<double, 3> q4 = {s, 0, -s};
    const std::array<double, 3> q5 = {s, -s, 0};
    const std::array<double, 3> q6 = {0, s, -s};
    const std::array<std::array<double, 3>, 6> q = {q1, q2, q3, q4, q5, q6};

    const double r2 = pow(64.0, 2);
    const double amplitude = amp_eq;
    const double rho = rho_seed;

    pfc::field::apply(m, get_field_name(), [=](const pfc::Real3 &X) {
      const double x = X[0];
      const double y = X[1];
      const double z = X[2];
      if (x * x + y * y + z * z >= r2) {
        return 0.0; // Outside seed: leave as zero
      }
      double u = rho;
      for (int qi = 0; qi < 6; ++qi) {
        u += 2.0 * amplitude * std::cos(q[qi][0] * x + q[qi][1] * y + q[qi][2] * z);
      }
      return u;
    });
  }
};

} // namespace pfc

#endif // PFC_INITIAL_CONDITIONS_SINGLE_SEED_HPP
