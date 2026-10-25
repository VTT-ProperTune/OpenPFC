// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include "eigensystem.hpp"

namespace singular_values {
// Example-owned one-sided complex Jacobi reference. Rotate columns directly;
// forming A* A would square the condition number and lose small values.
inline double smallest(const eigen_reference::Matrix &a) {
  using eigen_reference::Complex;
  const int n = static_cast<int>(a.size());
  if (n < 1 || n > 100) throw std::invalid_argument("Use square sizes in [1,100]");
  double scale = 0;
  for (const auto &row : a) {
    if (static_cast<int>(row.size()) != n)
      throw std::invalid_argument("Singular-value reference needs a square matrix");
    for (const auto value : row) {
      if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        throw std::invalid_argument("Singular-value input must be finite");
      scale = std::max(scale, std::abs(value));
    }
  }
  if (scale == 0) return 0;
  eigen_reference::Matrix columns(n, std::vector<Complex>(n));
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) columns[j][i] = a[i][j] / scale;
  const double tolerance = 32 * std::numeric_limits<double>::epsilon();
  // start snippet jacobi
  for (int sweep = 0; sweep < 100; ++sweep) {
    bool rotated = false;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q) {
        double alpha = 0, beta = 0;
        Complex gamma = 0;
        for (int i = 0; i < n; ++i) {
          alpha += std::norm(columns[p][i]);
          beta += std::norm(columns[q][i]);
          gamma += std::conj(columns[p][i]) * columns[q][i];
        }
        const double overlap = std::abs(gamma);
        if (overlap <= tolerance * std::sqrt(alpha) * std::sqrt(beta)) continue;
        const double tau = (beta - alpha) / (2 * overlap);
        const double t =
            std::copysign(1.0, tau) / (std::abs(tau) + std::hypot(1.0, tau));
        const double c = 1 / std::sqrt(1 + t * t);
        const Complex s = c * t * gamma / overlap;
        for (int i = 0; i < n; ++i) {
          const Complex left = columns[p][i], right = columns[q][i];
          columns[p][i] = c * left - std::conj(s) * right;
          columns[q][i] = s * left + c * right;
        }
        rotated = true;
      }
    if (!rotated) {
      double minimum = std::numeric_limits<double>::infinity();
      for (const auto &column : columns) {
        double norm2 = 0;
        for (auto value : column) norm2 += std::norm(value);
        minimum = std::min(minimum, std::sqrt(norm2));
      }
      return scale * minimum;
    }
  }
  // end snippet jacobi
  throw std::runtime_error("One-sided Jacobi iteration limit");
}
} // namespace singular_values
