// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../collocation.hpp"
#include <iostream>

int main(int argc, char **argv) {
  try {
    const int n = polynomial::degree(polynomial::input(argc, argv));
    const auto x = polynomial::nodes(n);
    const auto d = polynomial::differentiation(x);
    const auto d2 = polynomial::square(d);
    auto output = polynomial::output("solutions.csv");
    output << "boundary,kind,x,u,exact\n";
    // start snippet boundaries
    for (const std::string boundary : {"zero", "lifted", "mixed"}) {
      const bool mixed = boundary == "mixed";
      const int count = mixed ? n : n - 1;
      polynomial::Matrix a(count, polynomial::Vector(count));
      polynomial::Vector f(count);
      for (int row = 0; row < count; ++row) {
        const int i = row + 1;
        const auto &equation = mixed && i == n ? d[i] : d2[i];
        f[row] = mixed && i == n ? 0 : std::exp(4 * x[i]);
        for (int col = 0; col < count; ++col) a[row][col] = equation[col + 1];
      }
      const auto interior = polynomial::solve(a, f);
      polynomial::Vector u(n + 1);
      std::copy(interior.begin(), interior.end(), u.begin() + 1);
      if (boundary == "lifted")
        for (int j = 0; j <= n; ++j) u[j] += (x[j] + 1) / 2;
      // end snippet boundaries
      const polynomial::Interpolant interpolate(x, u);
      auto exact = [&](double point) {
        if (mixed)
          return (std::exp(4 * point) - 4 * std::exp(-4.0) * (point - 1) -
                  std::exp(4.0)) /
                 16;
        return (std::exp(4 * point) - std::sinh(4.0) * point - std::cosh(4.0)) / 16 +
               (boundary == "lifted" ? (point + 1) / 2 : 0);
      };
      for (int j = 0; j <= n; ++j)
        output << boundary << ",node," << x[j] << ',' << u[j] << ',' << exact(x[j])
               << '\n';
      for (int j = 0; j <= 400; ++j) {
        const double point = -1 + .005 * j;
        output << boundary << ",curve," << point << ',' << interpolate(point) << ','
               << exact(point) << '\n';
      }
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
