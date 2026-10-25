// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../collocation.hpp"
#include <iostream>

int main(int argc, char **argv) {
  try {
    const int maximum = polynomial::degree(polynomial::input(argc, argv), 50);
    auto output = polynomial::output("integrals.csv");
    output << "n,profile,integral,exact,error\n";
    // start snippet integrate
    for (int n = 1; n <= maximum; ++n) {
      const auto x = polynomial::nodes(n);
      const auto d = polynomial::differentiation(x);
      polynomial::Matrix a(n, polynomial::Vector(n));
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) a[i][j] = d[i][j];
      for (int type = 0; type < 4; ++type) {
        polynomial::Vector f;
        for (int j = 0; j < n; ++j)
          f.push_back(polynomial::profile(type, x[j]).value);
        const auto antiderivative = polynomial::solve(a, f);
        const double exact = polynomial::profile(type, 0).integral;
        output << n << ',' << type << ',' << antiderivative[0] << ',' << exact << ','
               << std::abs(antiderivative[0] - exact) << '\n';
      }
    }
    // end snippet integrate
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
