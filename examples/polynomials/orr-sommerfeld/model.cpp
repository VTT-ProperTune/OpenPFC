// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../clamped.hpp"
#include "../pencil.hpp"
#include <iostream>

int main(int argc, char **argv) {
  try {
    const auto input = polynomial::input(argc, argv);
    const auto degrees = input.value("degrees", std::vector<int>{40, 60, 80, 100});
    const double reynolds = input.value("reynolds", 5772.0);
    if (!std::isfinite(reynolds) || reynolds < 100 || reynolds > 1e6)
      throw std::invalid_argument("Use reynolds in [100,1000000]");
    if (degrees.empty() || degrees.size() > 8)
      throw std::invalid_argument("Use one through eight degrees");
    for (int n : degrees)
      if (n < 12 || n > 100 || n % 2)
        throw std::invalid_argument("Use even degrees in [12,100]");
    auto spectrum = polynomial::output("spectrum.csv");
    spectrum << "n,mode,real,imag,pencil_residual,transformed_residual\n";
    auto vectors = polynomial::output("vectors.csv");
    vectors << "n,mode,j,y,real,imag\n";
    auto growth = polynomial::output("growth.csv");
    growth << "n,reynolds,real,imag\n";
    for (int n : degrees) {
      const auto op = clamped::operators(n);
      const int m = n - 1;
      polynomial::Matrix b(m, polynomial::Vector(m));
      pencil::Matrix a(m, std::vector<pencil::Complex>(m));
      constexpr pencil::Complex imaginary(0, 1);
      // start snippet operators
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j) {
          const double identity = i == j ? 1 : 0;
          const double base_velocity = 1 - op.x[i + 1] * op.x[i + 1];
          b[i][j] = op.laplacian[i][j] - identity;
          a[i][j] =
              (op.biharmonic[i][j] - 2 * op.laplacian[i][j] + identity) / reynolds -
              imaginary * (2 * identity + base_velocity * b[i][j]);
        }
      // end snippet operators
      // start snippet solve
      const auto evolution = pencil::mass_solve(b, a);
      const auto modes = eigen_reference::solve(evolution);
      const auto leading = modes.back().value;
      // end snippet solve
      growth << n << ',' << reynolds << ',' << leading.real() << ','
             << leading.imag() << '\n';
      for (int index = 0; index < m; ++index) {
        const auto &mode = modes[index];
        const double residual = pencil::residual(a, b, mode);
        if (!std::isfinite(residual) || residual > 1e-9)
          throw std::runtime_error("Generalized eigenvector residual is too large");
        spectrum << n << ',' << index + 1 << ',' << mode.value.real() << ','
                 << mode.value.imag() << ',' << residual << ',' << mode.residual
                 << '\n';
        for (int j = 0; j <= n; ++j) {
          const pencil::Complex value = j == 0 || j == n ? 0 : mode.vector[j - 1];
          vectors << n << ',' << index + 1 << ',' << j << ',' << op.x[j] << ','
                  << value.real() << ',' << value.imag() << '\n';
        }
      }
      std::cout << "N = " << n << ", largest growth rate = " << leading.real()
                << '\n';
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
