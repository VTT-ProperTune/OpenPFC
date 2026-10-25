// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include "clamped.hpp"
#include "eigensystem.hpp"
#include <array>
#include <numeric>

namespace plate {
using polynomial::Matrix;
using polynomial::Vector;

// Interior arrays use x-fastest order: (i,j) -> i + m*j.
inline Vector apply(const clamped::Operators &op, const Vector &u) {
  const int m = static_cast<int>(op.laplacian.size());
  Vector result(m * m), xx(m * m);
  // start snippet tensor
  for (int j = 0; j < m; ++j)
    for (int i = 0; i < m; ++i)
      for (int k = 0; k < m; ++k) {
        result[i + m * j] +=
            op.biharmonic[i][k] * u[k + m * j] + op.biharmonic[j][k] * u[i + m * k];
        xx[i + m * j] += op.laplacian[i][k] * u[k + m * j];
      }
  for (int j = 0; j < m; ++j)
    for (int i = 0; i < m; ++i)
      for (int k = 0; k < m; ++k)
        result[i + m * j] += 2 * op.laplacian[j][k] * xx[i + m * k];
  // end snippet tensor
  return result;
}

// Orthonormal nodal bases for even (+1) or odd (-1) reflection.
inline Matrix parity_basis(int m, int sign) {
  Matrix basis;
  for (int i = 0; i < m / 2; ++i) {
    Vector v(m);
    v[i] = 1 / std::sqrt(2.0);
    v[m - 1 - i] = sign / std::sqrt(2.0);
    basis.push_back(std::move(v));
  }
  if (m % 2 && sign == 1) {
    Vector v(m);
    v[m / 2] = 1;
    basis.push_back(std::move(v));
  }
  return basis;
}

struct Mode {
  double eigenvalue;
  Vector u;
  int parity_x, parity_y, exchange;
};

// Same-parity sectors also split by u(y,x) = exchange*u(x,y).
// Cross-parity sectors are exchanged partners; solve one and transpose it.
inline std::vector<Mode> modes(const clamped::Operators &op) {
  const int m = static_cast<int>(op.laplacian.size());
  std::vector<Mode> result;
  for (const auto sector :
       {std::array{1, 1, 1}, std::array{1, 1, -1}, std::array{-1, -1, 1},
        std::array{-1, -1, -1}, std::array{1, -1, 0}}) {
    const auto [px, py, exchange] = sector;
    const auto bx = parity_basis(m, px), by = parity_basis(m, py);
    Matrix basis;
    for (int a = 0; a < static_cast<int>(bx.size()); ++a)
      for (int b = 0; b < static_cast<int>(by.size()); ++b) {
        if (exchange && (a > b || (a == b && exchange == -1))) continue;
        Vector v(m * m);
        for (int j = 0; j < m; ++j)
          for (int i = 0; i < m; ++i) {
            v[i + m * j] = bx[a][i] * by[b][j];
            if (exchange && a != b)
              v[i + m * j] =
                  (v[i + m * j] + exchange * bx[b][i] * by[a][j]) / std::sqrt(2.0);
          }
        basis.push_back(std::move(v));
      }
    const int size = static_cast<int>(basis.size());
    Matrix reduced(size, Vector(size));
    // start snippet symmetry
    for (int column = 0; column < size; ++column) {
      const auto image = plate::apply(op, basis[column]);
      for (int row = 0; row < size; ++row)
        reduced[row][column] = std::inner_product(
            basis[row].begin(), basis[row].end(), image.begin(), 0.0);
    }
    const auto eigenpairs =
        eigen_reference::solve(eigen_reference::complex_matrix(reduced));
    // end snippet symmetry
    for (const auto &pair : eigenpairs) {
      if (pair.value.real() <= 0 ||
          std::abs(pair.value.imag()) > 1e-8 * pair.value.real())
        throw std::runtime_error("Expected a real positive plate eigenvalue");
      Vector u(m * m);
      for (int k = 0; k < size; ++k)
        for (int j = 0; j < m * m; ++j) u[j] += basis[k][j] * pair.vector[k].real();
      double peak = 0;
      for (double value : u)
        if (std::abs(value) > std::abs(peak)) peak = value;
      if (!std::isfinite(peak) || peak == 0)
        throw std::runtime_error("Cannot normalize a zero plate mode");
      for (auto &value : u) value /= peak;
      result.push_back({pair.value.real(), u, px, py, exchange});
      if (!exchange) {
        Vector partner(m * m);
        for (int j = 0; j < m; ++j)
          for (int i = 0; i < m; ++i) partner[i + m * j] = u[j + m * i];
        result.push_back({pair.value.real(), std::move(partner), py, px, 0});
      }
    }
  }
  std::stable_sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
    return a.eigenvalue < b.eigenvalue;
  });
  return result;
}

// Values and slopes of the same weighted basis used in the beam chapter.
struct Evaluation {
  Matrix value, slope;
};

inline Evaluation evaluation(const Vector &x, const Vector &points) {
  const int n = static_cast<int>(x.size()) - 1;
  Evaluation result{Matrix(points.size(), Vector(n - 1)),
                    Matrix(points.size(), Vector(n - 1))};
  for (int j = 1; j < n; ++j) {
    Vector unit(n + 1);
    unit[j] = 1;
    const polynomial::Interpolant cardinal(x, unit);
    const polynomial::Interpolant derivative(x,
                                             pfc::fft::chebyshev_derivative(unit));
    for (std::size_t k = 0; k < points.size(); ++k) {
      const double point = points[k], w = 1 - point * point;
      result.value[k][j - 1] = w * cardinal(point) / (1 - x[j] * x[j]);
      result.slope[k][j - 1] =
          (-2 * point * cardinal(point) + w * derivative(point)) / (1 - x[j] * x[j]);
    }
  }
  return result;
}
} // namespace plate
