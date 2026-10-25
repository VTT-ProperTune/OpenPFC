// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../eigensystem.hpp"
#include <iostream>

int main(int argc, char **argv) {
  try {
    const auto input = polynomial::input(argc, argv);
    const int n = polynomial::degree(input, 36);
    const auto modes = input.value("modes", std::vector<int>{5, 10, 15, 20, 25, 30});
    for (int mode : modes)
      if (mode < 1 || mode >= n)
        throw std::invalid_argument("Modes must be in1..n-1");
    const auto x = polynomial::nodes(n);
    const auto d2 = polynomial::square(polynomial::differentiation(x));
    polynomial::Matrix a(n - 1, polynomial::Vector(n - 1));
    // start snippet eigenproblem
    for (int i = 1; i < n; ++i)
      for (int j = 1; j < n; ++j) a[i - 1][j - 1] = -d2[i][j];
    const auto pairs = eigen_reference::solve(eigen_reference::complex_matrix(a));
    auto spectrum = polynomial::output("spectrum.csv");
    spectrum << "mode,eigenvalue,imag,continuum,residual\n";
    for (int j = 0; j < n - 1; ++j)
      spectrum << j + 1 << ',' << pairs[j].value.real() << ','
               << pairs[j].value.imag() << ','
               << std::pow((j + 1) * std::numbers::pi / 2, 2) << ','
               << pairs[j].residual << '\n';
    // end snippet eigenproblem
    auto vectors = polynomial::output("vectors.csv");
    vectors << "mode,i,x,real,imag\n";
    for (int mode = 0; mode < n - 1; ++mode)
      for (int i = 0; i <= n; ++i) {
        const auto value = i == 0 || i == n ? eigen_reference::Complex{}
                                            : pairs[mode].vector[i - 1];
        vectors << mode + 1 << ',' << i << ',' << x[i] << ',' << value.real() << ','
                << value.imag() << '\n';
      }
    auto output = polynomial::output("modes.csv");
    output << "mode,kind,x,u\n";
    for (int mode : modes) {
      polynomial::Vector u(n + 1);
      for (int j = 1; j < n; ++j) u[j] = pairs[mode - 1].vector[j - 1].real();
      const polynomial::Interpolant interpolate(x, u);
      for (int j = 0; j <= n; ++j)
        output << mode << ",node," << x[j] << ',' << u[j] << '\n';
      for (int j = 0; j <= 400; ++j) {
        const double p = -1 + .005 * j;
        output << mode << ",curve," << p << ',' << interpolate(p) << '\n';
      }
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
