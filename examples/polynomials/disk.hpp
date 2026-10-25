// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include "collocation.hpp"
#include <openpfc/kernel/fft/chebyshev.hpp>

// Example-owned polar composition of native complete-domain derivatives/FFT.
namespace disk {
using polynomial::Matrix;
using polynomial::Vector;
using Complex = std::complex<double>;
using Coefficients = std::vector<pfc::fft::ComplexVector>;

struct Grid {
  int n, m, radial;
  Vector r;
  Matrix first, second;

  Grid(int degree, int angles) : n(degree), m(angles), radial(0) {
    if (n < 7 || n > 63 || n % 2 == 0 || m < 8 || m > 64 || m % 2)
      throw std::invalid_argument("Use odd n in [7,63] and even m in [8,64]");
    radial = (n - 1) / 2;
    // start snippet derivatives
    r = pfc::fft::chebyshev_lobatto(n);
    first.assign(n + 1, Vector(n + 1));
    Vector unit(n + 1);
    for (int j = 0; j <= n; ++j) {
      unit.at(j) = 1;
      const auto column = pfc::fft::chebyshev_derivative(unit);
      for (int i = 0; i <= n; ++i) first[i][j] = column[i];
      unit[j] = 0;
    }
    second = polynomial::square(first);
    // end snippet derivatives
  }

  double angle(int j) const { return 2 * std::numbers::pi * (j + 1) / m; }

  Matrix laplacian(int k) const {
    Matrix a(radial, Vector(radial));
    // start snippet radial
    const double parity = k % 2 ? -1 : 1;
    for (int i = 1; i <= radial; ++i)
      for (int j = 1; j <= radial; ++j) {
        a[i - 1][j - 1] = second[i][j] + first[i][j] / r[i] +
                          parity * (second[i][n - j] + first[i][n - j] / r[i]);
        if (i == j) a[i - 1][j - 1] -= k * k / (r[i] * r[i]);
      }
    // end snippet radial
    return a;
  }

  // Angular-fastest owned interior layout: theta_index + m*radial_index.
  Vector apply(const Vector &u) const {
    Vector result(u.size());
    pfc::fft::LocalFFT fft({m, 1, 1});
    pfc::fft::RealVector line(m), derivative(m);
    pfc::fft::ComplexVector modes(m / 2 + 1);
    for (int i = 0; i < radial; ++i) {
      for (int t = 0; t < m; ++t) line[t] = u[t + m * i];
      fft.forward(line, modes);
      for (int k = 0; k <= m / 2; ++k) modes[k] *= -k * k;
      fft.backward(modes, derivative);
      for (int t = 0; t < m; ++t) {
        result[t + m * i] = derivative[t] / (r[i + 1] * r[i + 1]);
        for (int j = 1; j <= radial; ++j) {
          const int opposite = (t + m / 2) % m;
          result[t + m * i] +=
              (second[i + 1][j] + first[i + 1][j] / r[i + 1]) * u[t + m * (j - 1)] +
              (second[i + 1][n - j] + first[i + 1][n - j] / r[i + 1]) *
                  u[opposite + m * (j - 1)];
        }
      }
    }
    return result;
  }
};

inline Vector extend(const Grid &grid, const Vector &positive, int k) {
  Vector values(grid.n + 1);
  for (int j = 1; j <= grid.radial; ++j) {
    values[j] = positive[j - 1];
    values[grid.n - j] = (k % 2 ? -1 : 1) * positive[j - 1];
  }
  return values;
}

// Reconstruct Fourier coefficients with the same signed-radius parity.
// The angular samples begin at dtheta, rather than zero.
class Interpolant {
  int m_;
  std::vector<polynomial::Interpolant> real_, imaginary_;

public:
  Interpolant(const Grid &grid, const Coefficients &coefficients) : m_(grid.m) {
    for (int k = 0; k <= m_ / 2; ++k) {
      Vector re(grid.radial), im(grid.radial);
      for (int j = 0; j < grid.radial; ++j) {
        re[j] = coefficients[j][k].real();
        im[j] = coefficients[j][k].imag();
      }
      real_.emplace_back(grid.r, extend(grid, re, k));
      imaginary_.emplace_back(grid.r, extend(grid, im, k));
    }
  }

  double operator()(double radius, double theta) const {
    double value = real_[0](radius);
    // At the display pole use the continuous, axisymmetric limit.
    if (radius == 0) return value / m_;
    for (int k = 1; k <= m_ / 2; ++k) {
      const Complex coefficient(real_[k](radius), imaginary_[k](radius));
      const double weight = k == m_ / 2 ? 1 : 2;
      value += weight *
               std::real(coefficient *
                         std::polar(1.0, k * (theta - 2 * std::numbers::pi / m_)));
    }
    return value / m_;
  }
};
} // namespace disk
