// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef PFC_INITIAL_CONDITIONS_RANDOM_SEEDS_HPP
#define PFC_INITIAL_CONDITIONS_RANDOM_SEEDS_HPP

#include <random>

#include "../field_modifier.hpp"
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
    const World &w = m.get_world();
    const Decomposition &decomp = m.get_decomposition();
    Field &field = m.get_real_field(get_field_name());
    Vec3<int> low = decomp.get_inbox().low;
    Vec3<int> high = decomp.get_inbox().high;

    // Use the new World API to get spacing and origin
    auto spacing = get_spacing(w);
    auto origin = get_origin(w);
    auto dx = spacing[0];
    auto dy = spacing[1];
    auto dz = spacing[2];
    auto x0 = origin[0];
    auto y0 = origin[1];
    auto z0 = origin[2];

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
    typedef std::array<double, 3> vec3;
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
            }
          }
          idx += 1;
        }
      }
    }
  }
};

} // namespace pfc

#endif // PFC_INITIAL_CONDITIONS_RANDOM_SEEDS_HPP
