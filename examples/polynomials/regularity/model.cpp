// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../collocation.hpp"
#include <iostream>

int main(int argc, char **argv) {
  try {
    const int maximum = polynomial::degree(polynomial::input(argc, argv), 50);
    auto output = polynomial::output("regularity.csv");
    output << "n,profile,error\n";
    // start snippet convergence
    for (int n = 1; n <= maximum; ++n) {
      const auto x = polynomial::nodes(n);
      const auto d = polynomial::differentiation(x);
      for (int type = 0; type < 4; ++type) {
        polynomial::Vector values;
        for (double point : x)
          values.push_back(polynomial::profile(type, point).value);
        const auto derivative = polynomial::apply(d, values);
        double error = 0;
        for (int j = 0; j <= n; ++j)
          error =
              std::max(error, std::abs(derivative[j] -
                                       polynomial::profile(type, x[j]).derivative));
        output << n << ',' << type << ',' << error << '\n';
      }
    }
    // end snippet convergence
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
