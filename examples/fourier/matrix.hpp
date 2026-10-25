// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <openpfc/kernel/fft/local_fft.hpp>
#include <stdexcept>
#include <vector>

namespace fourier_matrix {
using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;
// start snippet laplacian
inline Matrix negative_laplacian(int n, double length) {
  pfc::fft::LocalFFT fft({n, 1, 1});
  Matrix a(n, Vector(n));
  for (int column = 0; column < n; ++column) {
    Vector basis(n), image(n);
    basis[column] = 1;
    pfc::fft::ComplexVector coefficients(n / 2 + 1);
    fft.forward(basis, coefficients);
    for (int k = 0; k <= n / 2; ++k) {
      const double wave = 2 * std::numbers::pi * k / length;
      coefficients[k] *= wave * wave;
    }
    fft.backward(coefficients, image);
    for (int row = 0; row < n; ++row) a[row][column] = image[row];
  }
  return a;
}
// end snippet laplacian

// Cyclic symmetric Jacobi rotations, local to these small reference models.
inline Vector eigenvalues(Matrix a) {
  const int n = static_cast<int>(a.size());
  double scale = 1;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) scale = std::max(scale, std::abs(a[i][j]));
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off_diagonal = 0;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q) {
        off_diagonal = std::max(off_diagonal, std::abs(a[p][q]));
        if (std::abs(a[p][q]) < 8 * std::numeric_limits<double>::epsilon() * scale)
          continue;
        const double tau = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const double t =
            std::copysign(1.0 / (std::abs(tau) + std::hypot(1.0, tau)), tau);
        const double c = 1 / std::sqrt(1 + t * t), s = t * c;
        a[p][p] -= t * a[p][q];
        a[q][q] += t * a[p][q];
        a[p][q] = a[q][p] = 0;
        for (int j = 0; j < n; ++j) {
          if (j == p || j == q) continue;
          const double x = a[j][p], y = a[j][q];
          a[j][p] = a[p][j] = c * x - s * y;
          a[j][q] = a[q][j] = s * x + c * y;
        }
      }
    if (off_diagonal < 16 * std::numeric_limits<double>::epsilon() * scale) {
      Vector values(n);
      for (int i = 0; i < n; ++i) values[i] = a[i][i];
      std::sort(values.begin(), values.end());
      return values;
    }
  }
  throw std::runtime_error("Symmetric Jacobi iteration limit");
}
} // namespace fourier_matrix
