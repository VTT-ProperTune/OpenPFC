// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>

// start snippet sinc
double sinc(double x) {
  if (std::abs(x) < 1e-14) return 1;
  const double angle = std::numbers::pi * x;
  return std::sin(angle) / angle;
}

double interpolate(const pfc::data::Field<double> &samples, double x) {
  double value = 0;
  samples.for_each_owned([&](double node, double, double, double sample) {
    value += sample * sinc(x - node); // The sample spacing is h = 1.
  });
  return value;
}
// end snippet sinc

int main() {
  const auto domain = pfc::domain::create(
      pfc::GridSize{{21, 1, 1}}, pfc::PhysicalOrigin{{-10, 0, 0}},
      pfc::GridSpacing{{1, 1, 1}}, pfc::Bool3{{false, false, false}});
  pfc::data::Field<double> samples(domain, pfc::domain::index_box(domain), 0);
  std::filesystem::create_directories("results");
  std::ofstream output("results/interpolation.csv");
  output << std::setprecision(17) << "profile,x,value\n";
  double nodal_error = 0;
  // start snippet reconstruct
  for (int profile = 0; profile < 3; ++profile) {
    samples.apply([&](double x, double, double) {
      if (profile == 0) return x == 0 ? 1.0 : 0.0;
      if (profile == 1) return std::abs(x) <= 3 ? 1.0 : 0.0;
      return std::max(0.0, 1 - std::abs(x) / 3);
    });
    for (int point = 0; point <= 400; ++point) {
      const double x = -10.0 + point / 20.0;
      output << profile << ',' << x << ',' << interpolate(samples, x) << '\n';
    }
    samples.for_each_owned([&](double x, double, double, double sample) {
      nodal_error =
          std::max(nodal_error, std::abs(interpolate(samples, x) - sample));
    });
  }
  // end snippet reconstruct
  if (!output) throw std::runtime_error("Cannot write interpolation.csv");
  std::cout << "Maximum cardinal interpolation error at nodes=" << nodal_error
            << '\n';
  return nodal_error < 1e-13 ? 0 : 1;
}
