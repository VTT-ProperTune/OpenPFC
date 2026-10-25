// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../eigensystem.hpp"
#include <iostream>

int main(int argc, char **argv) {
  try {
    const auto input = polynomial::input(argc, argv);
    const auto sizes = input.value("sizes", std::vector<int>{12, 24, 36, 48});
    if (sizes.empty() || sizes.size() > 16)
      throw std::invalid_argument("Use1..16 sizes");
    auto output = polynomial::output("airy.csv");
    auto spectrum = polynomial::output("eigenvalues.csv");
    output << "n,kind,x,u\n";
    spectrum << "n,eigenvalue,imag,residual\n";
    for (int n : sizes) {
      polynomial::degree({{"n", n}});
      if (n < 12 || n % 2) throw std::invalid_argument("Use even n in [12,64]");
      const auto x = polynomial::nodes(n);
      const auto d2 = polynomial::square(polynomial::differentiation(x));
      const int center = n / 2;
      std::vector<int> free;
      for (int j = 1; j < n; ++j)
        if (j != center) free.push_back(j);
      polynomial::Matrix a(n - 2, polynomial::Vector(n - 2));
      // start snippet constraint
      for (int i = 0; i < n - 2; ++i)
        for (int j = 0; j < n - 2; ++j)
          a[i][j] =
              (d2[free[i]][free[j]] -
               d2[free[i]][center] * d2[center][free[j]] / d2[center][center]) /
              x[free[i]];
      const auto pairs = eigen_reference::solve(eigen_reference::complex_matrix(a));
      std::vector<eigen_reference::Eigenpair> positive;
      for (const auto &pair : pairs)
        if (pair.value.real() > 0) positive.push_back(pair);
      if (positive.size() < 5)
        throw std::runtime_error("Fewer than five positive eigenvalues");
      const auto &fifth = positive[4];
      polynomial::Vector u(n + 1);
      for (int j = 0; j < n - 2; ++j) u[free[j]] = fifth.vector[j].real();
      for (int j : free) u[center] -= d2[center][j] * u[j] / d2[center][center];
      const double ai0 = 1 / (std::pow(3.0, 2.0 / 3) * std::tgamma(2.0 / 3));
      const double scaling = ai0 / u[center];
      for (double &value : u) value *= scaling;
      // end snippet constraint
      spectrum << n << ',' << fifth.value.real() << ',' << fifth.value.imag() << ','
               << fifth.residual << '\n';
      const polynomial::Interpolant interpolate(x, u);
      for (int j = 0; j <= n; ++j)
        output << n << ",node," << x[j] << ',' << u[j] << '\n';
      for (int j = 0; j <= 400; ++j) {
        const double p = -1 + .005 * j;
        output << n << ",curve," << p << ',' << interpolate(p) << '\n';
      }
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
