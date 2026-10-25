// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../disk.hpp"
#include <iostream>

double exact(double r, double theta, bool manufactured) {
  if (manufactured)
    return (1 - r * r) *
           (1 + r * std::cos(theta) + .25 * r * r * std::cos(2 * theta));
  return 3 * (1 - std::pow(r, 4)) / 128 +
         (std::pow(r, 4) - r) * std::cos(theta) / 30 +
         (r * r - std::pow(r, 4)) * std::cos(2 * theta) / 96 +
         (std::pow(r, 6) - r * r) * std::sin(6 * theta) / 64 +
         (std::pow(r, 5) - r * r) * std::sin(5 * theta) / 84 +
         (std::pow(r, 7) - r * r) * std::sin(7 * theta) / 180;
}

int main(int argc, char **argv) {
  try {
    const auto input = polynomial::input(argc, argv);
    const disk::Grid grid(input.value("n", 31), input.value("m", 40));
    const bool manufactured = input.value("manufactured", false);
    if (!manufactured && grid.m < 16)
      throw std::invalid_argument(
          "Book forcing needs m >= 16 to resolve angular modes through seven");
    // start snippet source
    const auto source = [manufactured](double r, double theta) {
      return manufactured
                 ? -4 - 8 * r * std::cos(theta) - 3 * r * r * std::cos(2 * theta)
                 : -r * r * std::pow(std::sin(theta / 2), 4) +
                       std::sin(6 * theta) * std::pow(std::cos(theta / 2), 2);
    };
    // end snippet source
    pfc::fft::LocalFFT fft({grid.m, 1, 1});
    disk::Coefficients coefficients(grid.radial,
                                    pfc::fft::ComplexVector(grid.m / 2 + 1));
    disk::Vector forcing(grid.radial * grid.m), u(forcing.size());
    pfc::fft::RealVector line(grid.m);
    // start snippet transform
    for (int j = 0; j < grid.radial; ++j) {
      for (int t = 0; t < grid.m; ++t)
        forcing[t + grid.m * j] = line[t] = source(grid.r[j + 1], grid.angle(t));
      fft.forward(line, coefficients[j]);
    }
    // end snippet transform
    // start snippet solve
    for (int k = 0; k <= grid.m / 2; ++k) {
      const auto a = grid.laplacian(k);
      disk::Vector real_rhs(grid.radial), imaginary_rhs(grid.radial);
      for (int j = 0; j < grid.radial; ++j) {
        real_rhs[j] = coefficients[j][k].real();
        imaginary_rhs[j] = coefficients[j][k].imag();
      }
      const auto real = polynomial::solve(a, real_rhs);
      const auto imaginary = polynomial::solve(a, imaginary_rhs);
      for (int j = 0; j < grid.radial; ++j)
        coefficients[j][k] = disk::Complex(real[j], imaginary[j]);
    }
    for (int j = 0; j < grid.radial; ++j) {
      fft.backward(coefficients[j], line);
      for (int t = 0; t < grid.m; ++t) u[t + grid.m * j] = line[t];
    }
    // end snippet solve
    const auto image = grid.apply(u);
    double max_residual = 0, max_error = 0;
    for (int j = 0; j < grid.radial; ++j)
      for (int t = 0; t < grid.m; ++t) {
        max_residual = std::max(
            max_residual, std::abs(image[t + grid.m * j] - forcing[t + grid.m * j]));
        max_error = std::max(
            max_error, std::abs(u[t + grid.m * j] -
                                exact(grid.r[j + 1], grid.angle(t), manufactured)));
      }
    if (max_residual > 1e-8)
      throw std::runtime_error("Full polar Poisson residual is too large");
    auto diagnostics = polynomial::output("diagnostics.csv");
    diagnostics << "n,m,max_residual,max_error\n"
                << grid.n << ',' << grid.m << ',' << max_residual << ',' << max_error
                << '\n';
    const disk::Interpolant interpolate(grid, coefficients);
    auto output = polynomial::output("solution.csv");
    output << "kind,ir,it,r,theta,x,y,u,exact,f\n";
    const auto observe = [&](const char *kind, int ir, int it, double r,
                             double theta, double value) {
      output << kind << ',' << ir << ',' << it << ',' << r << ',' << theta << ','
             << r * std::cos(theta) << ',' << r * std::sin(theta) << ',' << value
             << ',' << exact(r, theta, manufactured) << ',' << source(r, theta)
             << '\n';
    };
    for (int ir = 0; ir <= grid.radial; ++ir)
      for (int it = 0; it < grid.m; ++it)
        observe("node", ir, it, grid.r[ir], grid.angle(it),
                ir == 0 ? 0 : u[it + grid.m * (ir - 1)]);
    for (int ir = 0; ir <= 64; ++ir)
      for (int it = 0; it <= 128; ++it) {
        const double r = ir / 64.0, theta = 2 * std::numbers::pi * it / 128;
        observe("curve", ir, it, r, theta, interpolate(r, theta));
      }
    std::cout << "Disk Poisson: max residual " << max_residual
              << ", max exact-field difference " << max_error << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
