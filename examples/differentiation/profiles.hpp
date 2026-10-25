// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <cmath>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string_view>

namespace profiles {

struct Sample {
  double value, derivative;
  bool differentiable = true;
};

// Independent analytic derivatives, rather than another differentiation scheme.
inline Sample evaluate(std::string_view name, double x) {
  const double s = std::sin(x), c = std::cos(x);
  if (name == "analytic") return {std::exp(s), c * std::exp(s)};
  if (name == "finite-smoothness")
    return {std::pow(std::abs(s), 3), 3 * s * c * std::abs(s)};
  if (name == "smooth-bump") {
    const double q = std::pow(std::sin(x / 2), 2);
    if (q < 1e-14) return {0, 0};
    const double value = std::exp(-1 / q);
    return {value, 0.5 * value * s / (q * q)};
  }
  if (name == "analytic-strip") {
    const double value = 1 / (1 + std::pow(std::sin(x / 2), 2));
    return {value, -std::sin(x / 2) * std::cos(x / 2) * value * value};
  }
  if (name == "band-limited") return {std::sin(10 * x), 10 * std::cos(10 * x)};
  if (name == "hat") {
    const double distance = x - std::numbers::pi;
    const bool corner =
        std::abs(distance) < 1e-12 || std::abs(std::abs(distance) - 2) < 1e-12;
    if (std::abs(distance) >= 2) return {0, 0, !corner};
    return {1 - std::abs(distance) / 2, distance < 0 ? 0.5 : -0.5, !corner};
  }
  throw std::invalid_argument("Unknown teaching profile");
}

// start snippet matrix
inline double matrix_derivative(std::span<const double> samples, int row) {
  const int n = static_cast<int>(samples.size());
  const double h = 2 * std::numbers::pi / n;
  double derivative = 0;
  for (int column = 0; column < n; ++column) {
    if (column == row) continue;
    const int difference = row - column;
    const double sign = difference % 2 == 0 ? 1.0 : -1.0;
    derivative += 0.5 * sign / std::tan(difference * h / 2) * samples[column];
  }
  return derivative;
}
// end snippet matrix

} // namespace profiles
