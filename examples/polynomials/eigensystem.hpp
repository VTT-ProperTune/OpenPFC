// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include "collocation.hpp"
#include <limits>
#include <numeric>

// Example-owned dense reference: bounded complex Hessenberg/shifted QR.
// OpenPFC's public API does not provide a dense eigensolver.
namespace eigen_reference {
using Complex = std::complex<double>;
using Matrix = std::vector<std::vector<Complex>>;
struct Eigenpair {
  Complex value;
  std::vector<Complex> vector;
  double residual;
};

inline Matrix complex_matrix(const polynomial::Matrix &a) {
  Matrix out(a.size(), std::vector<Complex>(a.size()));
  for (std::size_t i = 0; i < a.size(); ++i)
    for (std::size_t j = 0; j < a.size(); ++j) out[i][j] = a[i][j];
  return out;
}

inline std::vector<Eigenpair> solve(Matrix h) {
  const int n = static_cast<int>(h.size());
  if (n < 1 || n > 600)
    throw std::invalid_argument("Dense reference size must be 1..600");
  const Matrix original = h;
  Matrix q(n, std::vector<Complex>(n));
  double norm_a2 = 0;
  for (int i = 0; i < n; ++i) {
    if (static_cast<int>(h[i].size()) != n)
      throw std::invalid_argument("Matrix must be square");
    q[i][i] = 1;
    for (auto value : h[i]) norm_a2 += std::norm(value);
  }
  // Householder similarity reduction to upper Hessenberg form.
  for (int k = 0; k < n - 2; ++k) {
    std::vector<Complex> v(n);
    double norm2 = 0;
    for (int i = k + 1; i < n; ++i) {
      v[i] = h[i][k];
      norm2 += std::norm(v[i]);
    }
    if (norm2 == 0) continue;
    const Complex phase =
        std::abs(v[k + 1]) == 0 ? Complex(1) : v[k + 1] / std::abs(v[k + 1]);
    v[k + 1] += phase * std::sqrt(norm2);
    double length2 = 0;
    for (auto value : v) length2 += std::norm(value);
    for (auto &value : v) value /= std::sqrt(length2);
    for (int j = k; j < n; ++j) {
      Complex dot = 0;
      for (int i = k + 1; i < n; ++i) dot += std::conj(v[i]) * h[i][j];
      for (int i = k + 1; i < n; ++i) h[i][j] -= 2.0 * v[i] * dot;
    }
    for (int i = 0; i < n; ++i) {
      Complex dot_h = 0, dot_q = 0;
      for (int j = k + 1; j < n; ++j) {
        dot_h += h[i][j] * v[j];
        dot_q += q[i][j] * v[j];
      }
      for (int j = k + 1; j < n; ++j) {
        h[i][j] -= 2.0 * dot_h * std::conj(v[j]);
        q[i][j] -= 2.0 * dot_q * std::conj(v[j]);
      }
    }
    for (int i = k + 2; i < n; ++i) h[i][k] = 0;
  }
  constexpr double epsilon = 64 * std::numeric_limits<double>::epsilon();
  int active = n, iterations = 0;
  // start snippet qr
  while (active > 1) {
    const int last = active - 1;
    const double scale = std::abs(h[last][last]) + std::abs(h[last - 1][last - 1]);
    if (std::abs(h[last][last - 1]) <= epsilon * scale + 1e-290) {
      h[last][last - 1] = 0;
      --active;
      iterations = 0;
      continue;
    }
    if (++iterations > 4000) throw std::runtime_error("Dense QR iteration limit");
    int start = last - 1;
    while (start > 0) {
      if (std::abs(h[start][start - 1]) <=
          epsilon * (std::abs(h[start][start]) + std::abs(h[start - 1][start - 1])) +
              1e-290) {
        h[start][start - 1] = 0;
        break;
      }
      --start;
    }
    const Complex a = h[last - 1][last - 1], d = h[last][last];
    const Complex delta = (a - d) * .5;
    const Complex root =
        std::sqrt(delta * delta + h[last - 1][last] * h[last][last - 1]);
    const Complex first = (a + d) * .5 + root, second = (a + d) * .5 - root;
    Complex shift = std::abs(first - d) < std::abs(second - d) ? first : second;
    if (iterations % 30 == 0)
      shift = d + Complex(.75, .25) * std::abs(h[last][last - 1]);
    for (int i = start; i < active - 1; ++i) {
      const Complex x = i == start ? h[i][i] - shift : h[i][i - 1];
      const Complex y = i == start ? h[i + 1][i] : h[i + 1][i - 1];
      const double length = std::hypot(std::abs(x), std::abs(y));
      if (length == 0) continue;
      const double c = std::abs(x) / length;
      const Complex phase = std::abs(x) == 0 ? Complex(1) : x / std::abs(x);
      const Complex s = phase * std::conj(y) / length;
      for (int j = std::max(0, i - 1); j < n; ++j) {
        const Complex upper = h[i][j], lower = h[i + 1][j];
        h[i][j] = c * upper + s * lower;
        h[i + 1][j] = -std::conj(s) * upper + c * lower;
      }
      for (int row = 0; row < n; ++row) {
        const Complex left = h[row][i], right = h[row][i + 1];
        h[row][i] = c * left + std::conj(s) * right;
        h[row][i + 1] = -s * left + c * right;
        const Complex qleft = q[row][i], qright = q[row][i + 1];
        q[row][i] = c * qleft + std::conj(s) * qright;
        q[row][i + 1] = -s * qleft + c * qright;
      }
      if (i > start) h[i + 1][i - 1] = 0;
    }
  }
  // end snippet qr
  std::vector<Eigenpair> result;
  for (int column = 0; column < n; ++column) {
    const Complex lambda = h[column][column];
    std::vector<Complex> y(n), v(n);
    y[column] = 1;
    for (int i = column - 1; i >= 0; --i) {
      Complex rhs = 0;
      for (int j = i + 1; j <= column; ++j) rhs += h[i][j] * y[j];
      Complex denominator = h[i][i] - lambda;
      if (std::abs(denominator) < epsilon * std::max(1.0, std::sqrt(norm_a2)))
        denominator = epsilon * std::max(1.0, std::sqrt(norm_a2));
      y[i] = -rhs / denominator;
      double size = 1;
      for (auto value : y) size = std::max(size, std::abs(value));
      if (size > 1e100)
        for (auto &value : y) value /= size;
    }
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) v[i] += q[i][j] * y[j];
    const auto peak = *std::max_element(v.begin(), v.end(), [](auto x, auto z) {
      return std::abs(x) < std::abs(z);
    });
    for (auto &value : v) value /= peak;
    double residual2 = 0, vector2 = 0;
    for (int i = 0; i < n; ++i) {
      Complex value = -lambda * v[i];
      for (int j = 0; j < n; ++j) value += original[i][j] * v[j];
      residual2 += std::norm(value);
      vector2 += std::norm(v[i]);
    }
    const double residual = std::sqrt(residual2 / vector2) /
                            std::max(1e-290, std::sqrt(norm_a2) + std::abs(lambda));
    if (!std::isfinite(residual) || residual > 1e-9)
      throw std::runtime_error("Dense eigenvector residual is too large");
    result.push_back({lambda, std::move(v), residual});
  }
  std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
    return a.value.real() < b.value.real();
  });
  return result;
}
} // namespace eigen_reference
