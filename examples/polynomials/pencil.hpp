// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include "eigensystem.hpp"

// Small example-owned reference for an invertible real mass matrix.
// This is not a public generalized eigensolver or a QZ implementation.
namespace pencil {
using eigen_reference::Complex;
using eigen_reference::Matrix;

inline Matrix mass_solve(polynomial::Matrix b, Matrix a) {
  const int n = static_cast<int>(b.size());
  if (n < 1 || n > 100 || static_cast<int>(a.size()) != n)
    throw std::invalid_argument("Mass solve needs matching sizes in [1,100]");
  for (int i = 0; i < n; ++i)
    if (static_cast<int>(a[i].size()) != n || static_cast<int>(b[i].size()) != n)
      throw std::invalid_argument("Mass solve needs square matrices");
  // start snippet mass
  for (int k = 0; k < n; ++k) {
    int pivot = k;
    for (int i = k + 1; i < n; ++i)
      if (std::abs(b[i][k]) > std::abs(b[pivot][k])) pivot = i;
    if (!std::isfinite(b[pivot][k]) || b[pivot][k] == 0)
      throw std::runtime_error("Singular mass matrix");
    std::swap(b[k], b[pivot]);
    std::swap(a[k], a[pivot]);
    for (int i = k + 1; i < n; ++i) {
      const double multiplier = b[i][k] / b[k][k];
      b[i][k] = 0;
      for (int j = k + 1; j < n; ++j) b[i][j] -= multiplier * b[k][j];
      for (int column = 0; column < n; ++column)
        a[i][column] -= multiplier * a[k][column];
    }
  }
  for (int i = n - 1; i >= 0; --i)
    for (int column = 0; column < n; ++column) {
      for (int j = i + 1; j < n; ++j) a[i][column] -= b[i][j] * a[j][column];
      a[i][column] /= b[i][i];
    }
  // end snippet mass
  return a;
}

inline double residual(const Matrix &a, const polynomial::Matrix &b,
                       const eigen_reference::Eigenpair &pair) {
  const int n = static_cast<int>(a.size());
  double error2 = 0, vector2 = 0, a2 = 0, b2 = 0;
  for (int i = 0; i < n; ++i) {
    Complex error = 0;
    for (int j = 0; j < n; ++j) {
      error += (a[i][j] - pair.value * b[i][j]) * pair.vector[j];
      a2 += std::norm(a[i][j]);
      b2 += b[i][j] * b[i][j];
    }
    error2 += std::norm(error);
    vector2 += std::norm(pair.vector[i]);
  }
  return std::sqrt(error2 / vector2) /
         (std::sqrt(a2) + std::abs(pair.value) * std::sqrt(b2));
}
} // namespace pencil
