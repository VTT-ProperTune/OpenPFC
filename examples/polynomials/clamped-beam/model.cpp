// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../clamped.hpp"
#include <iostream>

int main(int argc, char **argv) {
  try {
    const auto input = polynomial::input(argc, argv);
    const int n = polynomial::degree(input, 15);
    if (n < 4 || n > 32) throw std::invalid_argument("Use n in [4,32]");
    const auto operators = clamped::operators(n);
    const auto &x = operators.x;
    polynomial::Vector forcing(n - 1);
    for (int j = 1; j < n; ++j) forcing[j - 1] = std::exp(x[j]);
    // start snippet solve
    const auto interior = polynomial::solve(operators.biharmonic, forcing);
    polynomial::Vector auxiliary(n + 1);
    for (int j = 1; j < n; ++j) auxiliary[j] = interior[j - 1] / (1 - x[j] * x[j]);
    const polynomial::Interpolant interpolate(x, auxiliary);
    const polynomial::Interpolant derivative(
        x, pfc::fft::chebyshev_derivative(auxiliary));
    // end snippet solve
    const double sh = std::sinh(1.0), ch = std::cosh(1.0);
    const double a = ch - sh / 2, b = 1.5 * sh - ch / 2;
    const double c = sh / 2, d = (ch - sh) / 2;
    auto output = polynomial::output("beam.csv");
    output << "kind,x,u,exact,du,exact_du\n";
    const auto observe = [&](const char *kind, double point) {
      const double w = interpolate(point);
      const double u = (1 - point * point) * w;
      const double du = -2 * point * w + (1 - point * point) * derivative(point);
      const double exact =
          std::exp(point) - (a + point * (b + point * (c + point * d)));
      const double exact_du =
          std::exp(point) - (b + point * (2 * c + 3 * point * d));
      output << kind << ',' << point << ',' << u << ',' << exact << ',' << du << ','
             << exact_du << '\n';
    };
    for (double point : x) observe("node", point);
    for (int j = 0; j <= 400; ++j) observe("curve", -1 + j / 200.0);
    std::cout << "Clamped fourth-order problem, degree " << n << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
