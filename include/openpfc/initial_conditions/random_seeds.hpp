// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file random_seeds.hpp
 * @brief Random distribution of crystalline seeds initial condition
 *
 * @details
 * This file defines the RandomSeeds class, which places spherical crystalline
 * seeds at random locations throughout the domain. Useful for:
 * - Realistic polycrystalline microstructures
 * - Statistical studies of grain growth
 * - Homogeneous nucleation simulations
 *
 * The number and properties of seeds are configurable. Seeds are randomly
 * positioned using the Mersenne Twister random number generator.
 *
 * Usage:
 * @code
 * auto ic = std::make_unique<pfc::RandomSeeds>();
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

#ifndef PFC_INITIAL_CONDITIONS_RANDOM_SEEDS_HPP
#define PFC_INITIAL_CONDITIONS_RANDOM_SEEDS_HPP

#include <random>

#include "../field_modifier.hpp"
#include "openpfc/field/operations.hpp"
#include "seed.hpp"

namespace pfc {

class RandomSeeds : public FieldModifier {
  double m_density, m_amplitude;

public:
  void set_amplitude(double amplitude) { m_amplitude = amplitude; }
  double get_amplitude() const { return m_amplitude; }
  void set_density(double density) { m_density = density; }
  double get_density() const { return m_density; }

  void apply(Model &m, double) override {
    // Functional coordinate-space implementation using field::apply
    std::vector<Seed> seeds;
    const int nseeds = 150;
    const double radius = 20.0;
    const double lower_x = -128.0 + radius;
    const double upper_x = -128.0 + 3 * radius;
    const double lower_y = -128.0;
    const double upper_y = 128.0;
    const double lower_z = -128.0;
    const double upper_z = 128.0;

    std::mt19937_64 re(42);
    std::uniform_real_distribution<double> rx(lower_x, upper_x);
    std::uniform_real_distribution<double> ry(lower_y, upper_y);
    std::uniform_real_distribution<double> rz(lower_z, upper_z);
    std::uniform_real_distribution<double> ro(0.0, 8.0 * atan(1.0));
    using vec3 = std::array<double, 3>;
    auto random_location = [&re, &rx, &ry, &rz]() {
      return vec3({rx(re), ry(re), rz(re)});
    };
    auto random_orientation = [&re, &ro]() {
      return vec3({ro(re), ro(re), ro(re)});
    };

    for (int i = 0; i < nseeds; i++) {
      const std::array<double, 3> location = random_location();
      const std::array<double, 3> orientation = random_orientation();
      const Seed seed(location, orientation, radius, get_density(), get_amplitude());
      seeds.push_back(seed);
    }

    pfc::field::apply(m, get_field_name(), [seeds](const pfc::Real3 &X) {
      for (const auto &seed : seeds) {
        if (seed.is_inside(X)) {
          return seed.get_value(X);
        }
      }
      return 0.0; // Outside seeds
    });
  }
};

} // namespace pfc

#endif // PFC_INITIAL_CONDITIONS_RANDOM_SEEDS_HPP
