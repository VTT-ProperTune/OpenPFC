// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../collocation.hpp"
#include <iostream>

int main(int argc, char **argv) {
  try {
    const int n = polynomial::degree(polynomial::input(argc, argv));
    auto output = polynomial::output("interpolation.csv");
    output << "grid,kind,x,value,exact\n";
    // start snippet interpolate
    for (bool chebyshev : {false, true}) {
      const auto x = polynomial::nodes(n, chebyshev);
      polynomial::Vector values;
      for (double point : x) values.push_back(1 / (1 + 16 * point * point));
      const polynomial::Interpolant interpolate(x, values);
      const auto name = chebyshev ? "chebyshev" : "equispaced";
      for (int j = 0; j <= n; ++j)
        output << name << ",node," << x[j] << ',' << values[j] << ',' << values[j]
               << '\n';
      for (int j = 0; j <= 404; ++j) {
        const double point = -1.01 + .005 * j;
        output << name << ",curve," << point << ',' << interpolate(point) << ','
               << 1 / (1 + 16 * point * point) << '\n';
      }
    }
    // end snippet interpolate
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
