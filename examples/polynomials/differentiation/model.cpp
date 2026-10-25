// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../collocation.hpp"
#include <iostream>

int main(int argc, char **argv) {
  try {
    const auto config = polynomial::input(argc, argv);
    const auto sizes = config.value("sizes", std::vector<int>{10, 20});
    if (sizes.empty() || sizes.size() > 16)
      throw std::invalid_argument("Use 1..16 sizes");
    auto output = polynomial::output("derivatives.csv");
    output << "n,x,value,matrix,fourier,exact\n";
    // start snippet differentiate
    for (int n : sizes) {
      polynomial::degree({{"n", n}});
      const auto x = polynomial::nodes(n);
      polynomial::Vector values;
      for (double point : x) values.push_back(std::exp(point) * std::sin(5 * point));
      const auto dense = polynomial::apply(polynomial::differentiation(x), values);
      const auto transformed = polynomial::fft_derivative(values);
      for (int j = 0; j <= n; ++j) {
        const double exact =
            std::exp(x[j]) * (std::sin(5 * x[j]) + 5 * std::cos(5 * x[j]));
        output << n << ',' << x[j] << ',' << values[j] << ',' << dense[j] << ','
               << transformed[j] << ',' << exact << '\n';
      }
    }
    // end snippet differentiate
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
