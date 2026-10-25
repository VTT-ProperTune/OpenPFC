// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../collocation.hpp"
#include <iostream>

int main(int argc, char **argv) {
  try {
    const int n = polynomial::degree(polynomial::input(argc, argv));
    auto line = polynomial::output("nodal.csv");
    auto plane = polynomial::output("potential.csv");
    line << "grid,x,value\n";
    plane << "grid,real,imag,magnitude\n";
    // start snippet product
    for (bool chebyshev : {false, true}) {
      const auto nodes = polynomial::nodes(n, chebyshev);
      auto product = [&](std::complex<double> z) {
        std::complex<double> value = 1;
        for (double x : nodes) value *= z - x;
        return value;
      };
      const auto name = chebyshev ? "chebyshev" : "equispaced";
      for (int i = 0; i <= 400; ++i) {
        const double x = -1 + .005 * i;
        line << name << ',' << x << ',' << product(x).real() << '\n';
      }
      for (int j = 0; j <= 112; ++j)
        for (int i = 0; i <= 140; ++i) {
          const double x = -1.4 + .02 * i, y = -1.12 + .02 * j;
          plane << name << ',' << x << ',' << y << ',' << std::abs(product({x, y}))
                << '\n';
        }
    }
    // end snippet product
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
