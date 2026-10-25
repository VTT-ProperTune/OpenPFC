// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../eigensystem.hpp"
#include <iostream>

int main(int argc, char **argv) {
  try {
    const auto input = polynomial::input(argc, argv);
    const int n = polynomial::degree(input, 16), m = n - 1;
    if (n < 3 || n > 20)
      throw std::invalid_argument("Membrane dense reference uses n in [3,20]");
    const auto x = polynomial::nodes(n);
    const auto d2 = polynomial::square(polynomial::differentiation(x));
    polynomial::Matrix a(m * m, polynomial::Vector(m * m));
    // start snippet membrane
    for (int j = 0; j < m; ++j)
      for (int i = 0; i < m; ++i) {
        const int row = i + m * j;
        for (int k = 0; k < m; ++k) {
          a[row][k + m * j] -= d2[i + 1][k + 1];
          a[row][i + m * k] -= d2[j + 1][k + 1];
        }
        a[row][row] += std::exp(20 * (x[j + 1] - x[i + 1] - 1));
      }
    const auto modes = eigen_reference::solve(eigen_reference::complex_matrix(a));
    // end snippet membrane
    auto spectrum = polynomial::output("spectrum.csv");
    spectrum << "mode,eigenvalue,imag,residual\n";
    auto output = polynomial::output("modes.csv");
    output << "mode,i,j,x,y,u\n";
    for (int mode = 0; mode < 4; ++mode) {
      spectrum << mode + 1 << ',' << modes[mode].value.real() << ','
               << modes[mode].value.imag() << ',' << modes[mode].residual << '\n';
      for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i) {
          const double u = (i == 0 || j == 0 || i == n || j == n)
                               ? 0
                               : modes[mode].vector[i - 1 + m * (j - 1)].real();
          output << mode + 1 << ',' << i << ',' << j << ',' << x[i] << ',' << x[j]
                 << ',' << u << '\n';
        }
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
