#pragma once

#include <random>

#include "../field_modifier.hpp"
#include "seed.hpp"

namespace pfc {

class RandomSeeds : public FieldModifier {

public:
  double rho, amplitude, n0;

  void apply(Model &m, double) override {
    const World &w = m.get_world();
    const Decomposition &decomp = m.get_decomposition();
    Field &field = m.get_field();
    Vec3<int> low = decomp.inbox.low;
    Vec3<int> high = decomp.inbox.high;

    auto dx = w.dx;
    auto dy = w.dy;
    auto dz = w.dz;
    auto x0 = w.x0;
    auto y0 = w.y0;
    auto z0 = w.z0;

    std::vector<Seed> seeds;

    const int nseeds = 150;
    const double radius = 20.0;
    const double lower_x = -128.0 + radius;
    const double upper_x = -128.0 + 3 * radius;
    const double lower_y = -128.0;
    const double upper_y = 128.0;
    const double lower_z = -128.0;
    const double upper_z = 128.0;
    srand(42);
    std::uniform_real_distribution<double> rx(lower_x, upper_x);
    std::uniform_real_distribution<double> ry(lower_y, upper_y);
    std::uniform_real_distribution<double> rz(lower_z, upper_z);
    std::uniform_real_distribution<double> ro(0.0, 8.0 * atan(1.0));
    std::default_random_engine re;
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
      const Seed seed(location, orientation, radius, rho, amplitude);
      seeds.push_back(seed);
    }

    std::fill(field.begin(), field.end(), n0);
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
