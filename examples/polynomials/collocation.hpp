// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

// Small, example-owned Chebyshev reference. This is not a public OpenPFC API.
#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <numbers>
#include <openpfc/kernel/fft/local_fft.hpp>
#include <openpfc/solvers/krylov.hpp>
#include <span>
#include <stdexcept>
#include <vector>

namespace polynomial {
using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

inline nlohmann::json input(int argc, char **argv) {
  if (argc < 2) return nlohmann::json::object();
  std::ifstream file(argv[1]);
  if (!file) throw std::runtime_error("Cannot open input JSON");
  nlohmann::json value;
  file >> value;
  return value;
}
inline int degree(const nlohmann::json &config, int fallback = 16) {
  const int n = config.value("n", fallback);
  if (n < 2 || n > 64) throw std::invalid_argument("Use n in [2,64]");
  return n;
}
inline std::ofstream output(const char *name) {
  std::filesystem::create_directories("results");
  std::ofstream file(std::string("results/") + name);
  if (!file) throw std::runtime_error("Cannot open output CSV");
  file.exceptions(std::ios::badbit | std::ios::failbit);
  file << std::setprecision(17);
  return file;
}

// start snippet nodes
inline Vector nodes(int n, bool chebyshev = true) {
  Vector x(n + 1);
  for (int j = 0; j <= n; ++j)
    x[j] = chebyshev ? std::cos(std::numbers::pi * j / n) : -1.0 + 2.0 * j / n;
  return x;
}
// end snippet nodes

// start snippet barycentric
class Interpolant {
  Vector x_, values_, weights_;

public:
  Interpolant(Vector x, Vector values)
      : x_(std::move(x)), values_(std::move(values)), weights_(x_.size(), 1) {
    if (values_.size() != x_.size())
      throw std::invalid_argument("Sample sizes differ");
    for (std::size_t j = 0; j < x_.size(); ++j)
      for (std::size_t k = 0; k < x_.size(); ++k)
        if (j != k) weights_[j] /= x_[j] - x_[k];
  }
  double operator()(double point) const {
    double numerator = 0, denominator = 0;
    for (std::size_t j = 0; j < x_.size(); ++j) {
      if (point == x_[j]) return values_[j];
      const double term = weights_[j] / (point - x_[j]);
      numerator += term * values_[j];
      denominator += term;
    }
    return numerator / denominator;
  }
};
// end snippet barycentric

// start snippet matrix
inline Matrix differentiation(std::span<const double> x) {
  const int n = static_cast<int>(x.size()) - 1;
  Matrix d(n + 1, Vector(n + 1));
  for (int i = 0; i <= n; ++i)
    for (int j = 0; j <= n; ++j) {
      if (i == j) continue;
      const double ci = (i == 0 || i == n) ? 2 : 1;
      const double cj = (j == 0 || j == n) ? 2 : 1;
      d[i][j] = (ci / cj) * ((i + j) % 2 ? -1 : 1) / (x[i] - x[j]);
      d[i][i] -= d[i][j];
    }
  return d;
}
// end snippet matrix
inline Vector apply(const Matrix &a, std::span<const double> v) {
  Vector out(a.size());
  for (std::size_t i = 0; i < a.size(); ++i)
    for (std::size_t j = 0; j < v.size(); ++j) out[i] += a[i][j] * v[j];
  return out;
}
inline Matrix square(const Matrix &a) {
  Matrix out(a.size(), Vector(a.size()));
  for (std::size_t i = 0; i < a.size(); ++i)
    for (std::size_t k = 0; k < a.size(); ++k)
      for (std::size_t j = 0; j < a.size(); ++j) out[i][j] += a[i][k] * a[k][j];
  return out;
}

// start snippet cosine
inline Vector fft_derivative(const Vector &values) {
  const int n = static_cast<int>(values.size()) - 1;
  pfc::fft::LocalFFT fft({2 * n, 1, 1});
  Vector even(2 * n);
  for (int j = 0; j < 2 * n; ++j) even[j] = values[j <= n ? j : 2 * n - j];
  pfc::fft::ComplexVector coefficients(n + 1);
  fft.forward(even, coefficients);
  Vector a(n + 1), derivative(n + 1);
  for (int k = 0; k <= n; ++k)
    a[k] = coefficients[k].real() / ((k == 0 || k == n) ? 2 * n : n);
  // f(cos(theta)) = sum a_k cos(k theta). Differentiate that polynomial.
  for (int j = 0; j <= n; ++j) {
    const double theta = std::numbers::pi * j / n;
    for (int k = 1; k <= n; ++k) {
      const double basis = j == 0 ? k
                                  : (j == n ? (k % 2 ? k : -k)
                                            : std::sin(k * theta) / std::sin(theta));
      derivative[j] += k * a[k] * basis;
    }
  }
  return derivative;
}
// end snippet cosine

struct Sample {
  double value, derivative, integral;
};
inline Sample profile(int type, double x) {
  switch (type) {
  case 0: return {std::pow(std::abs(x), 3), 3 * x * std::abs(x), .5};
  case 1: {
    const double v = std::abs(x) < 1e-14 ? 0 : std::exp(-1 / (x * x));
    return {v, v == 0 ? 0 : 2 * v / (x * x * x),
            2 * (std::exp(-1.0) - std::sqrt(std::numbers::pi) * std::erfc(1.0))};
  }
  case 2:
    return {1 / (1 + x * x), -2 * x / std::pow(1 + x * x, 2), std::numbers::pi / 2};
  case 3: return {std::pow(x, 10), 10 * std::pow(x, 9), 2.0 / 11};
  default: throw std::invalid_argument("Unknown profile");
  }
}

// start snippet solve
inline Vector solve(Matrix a, Vector b) {
  // Row scaling changes the residual norm, not the solution.
  for (std::size_t i = 0; i < a.size(); ++i) {
    double scale = 0;
    for (double entry : a[i]) scale = std::max(scale, std::abs(entry));
    if (!(scale > 0)) throw std::runtime_error("Empty equation");
    for (double &entry : a[i]) entry /= scale;
    b[i] /= scale;
  }
  pfc::solvers::VectorOperations<> ops;
  pfc::solvers::KrylovControls controls;
  controls.relative_tolerance = 1e-12;
  controls.absolute_tolerance = 1e-14;
  controls.restart = static_cast<int>(b.size());
  controls.maximum_iterations = 4 * controls.restart;
  Vector u(b.size());
  const auto result = pfc::solvers::gmres(
      ops, [&](const auto &v, auto &w) { w = polynomial::apply(a, v); }, b, u,
      controls);
  if (!result.converged()) throw std::runtime_error("GMRES did not converge");
  return u;
}
// end snippet solve
} // namespace polynomial
