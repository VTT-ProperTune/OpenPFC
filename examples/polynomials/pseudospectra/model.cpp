// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../singular_values.hpp"
#include <iostream>
#include <openpfc/kernel/fft/chebyshev.hpp>

int main(int argc, char **argv) {
  try {
    const auto input = polynomial::input(argc, argv);
    const int n = input.value("n", 70);
    const double length = input.value("length", 6.0),
                 step = input.value("step", 2.0);
    if (n < 6 || n > 80) throw std::invalid_argument("Use n in [6,80]");
    if (!std::isfinite(length) || length < 2 || length > 12)
      throw std::invalid_argument("Use length in [2,12]");
    if (!std::isfinite(step) || step < .5 || step > 10)
      throw std::invalid_argument("Use step in [0.5,10]");
    const auto x = pfc::fft::chebyshev_lobatto(n);
    using eigen_reference::Complex;
    eigen_reference::Matrix a(n - 1, std::vector<Complex>(n - 1));
    // start snippet oscillator
    for (int j = 1; j < n; ++j) {
      polynomial::Vector unit(n + 1);
      unit[j] = 1;
      const auto first = pfc::fft::chebyshev_derivative(unit);
      const auto second = pfc::fft::chebyshev_derivative(first);
      for (int i = 1; i < n; ++i) a[i - 1][j - 1] = -second[i] / (length * length);
      a[j - 1][j - 1] += Complex(1, 3) * std::pow(length * x[j], 2);
    }
    const auto modes = eigen_reference::solve(a);
    // end snippet oscillator
    auto spectrum = polynomial::output("spectrum.csv");
    spectrum << "mode,real,imag,residual\n";
    for (int j = 0; j < n - 1; ++j)
      spectrum << j + 1 << ',' << modes[j].value.real() << ','
               << modes[j].value.imag() << ',' << modes[j].residual << '\n';
    auto output = polynomial::output("pseudospectrum.csv");
    output << "real,imag,sigma_min\n";
    // start snippet pseudospectrum
    for (int iy = 0; iy <= static_cast<int>(40 / step); ++iy)
      for (int ix = 0; ix <= static_cast<int>(50 / step); ++ix) {
        const Complex z(ix * step, iy * step);
        auto shifted = a;
        for (auto &row : shifted)
          for (auto &value : row) value = -value;
        for (int j = 0; j < n - 1; ++j) shifted[j][j] += z;
        output << z.real() << ',' << z.imag() << ','
               << singular_values::smallest(shifted) << '\n';
      }
    // end snippet pseudospectrum
    std::cout << "Davies oscillator, degree " << n << ", complex grid step " << step
              << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
