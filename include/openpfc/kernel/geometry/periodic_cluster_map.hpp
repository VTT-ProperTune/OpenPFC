// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file periodic_cluster_map.hpp
 * @brief Static periodic clustering map on a uniform computational line.
 *
 * This is the #210 feasibility prototype. `ξ` stays uniformly spaced and
 * the derivative is a Fourier multiplier on that line, through
 * `ComplexLineFFT`. The physical coordinate is
 *
 * `x(ξ) = ξ + (a L / 2π) sin(2π ξ / L)`, `x'(ξ) = 1 + a cos(2π ξ / L)`,
 *
 * with `|a| < 1` so the Jacobian stays positive. `a < 0` packs nodes at
 * `ξ = 0`. The physical derivative is `(du/dξ) / x'`, and the second
 * derivative repeats that step. The Nyquist multiplier is zero.
 *
 * `Domain` and HeFFTe are not used. A moving monitor is not this map.
 */

#include <openpfc/kernel/fft/complex_line_fft.hpp>

#include <cmath>
#include <complex>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

namespace pfc::geometry {

struct PeriodicClusterMap {
  double period = 1;
  double amplitude = 0;
};

[[nodiscard]] inline PeriodicClusterMap periodic_cluster_map(double period,
                                                             double amplitude) {
  if (!std::isfinite(period) || !std::isfinite(amplitude) || !(period > 0.0) ||
      !(std::fabs(amplitude) < 1.0)) {
    throw std::invalid_argument(
        "periodic cluster map needs a positive period and |amplitude| < 1");
  }
  return PeriodicClusterMap{period, amplitude};
}

[[nodiscard]] inline double cluster_computational(const PeriodicClusterMap &map,
                                                  int index, int count) {
  if (count < 2 || index < 0 || index >= count) {
    throw std::invalid_argument("periodic cluster index is outside the line");
  }
  return map.period * static_cast<double>(index) / static_cast<double>(count);
}

[[nodiscard]] inline double cluster_physical(const PeriodicClusterMap &map,
                                             double xi) {
  if (!std::isfinite(xi)) {
    throw std::invalid_argument("periodic cluster coordinate must be finite");
  }
  const double angle = 2.0 * std::numbers::pi * xi / map.period;
  return xi +
         (map.amplitude * map.period / (2.0 * std::numbers::pi)) * std::sin(angle);
}

[[nodiscard]] inline double cluster_jacobian(const PeriodicClusterMap &map,
                                             double xi) {
  if (!std::isfinite(xi)) {
    throw std::invalid_argument("periodic cluster coordinate must be finite");
  }
  const double angle = 2.0 * std::numbers::pi * xi / map.period;
  return 1.0 + map.amplitude * std::cos(angle);
}

namespace cluster_detail {

inline std::vector<double> xi_derivative(std::span<const double> samples,
                                         double period) {
  const int count = static_cast<int>(samples.size());
  if (count < 2) {
    throw std::invalid_argument(
        "periodic cluster derivative needs at least two nodes");
  }
  pfc::fft::ComplexLineFFT fft(count);
  std::vector<std::complex<double>> input(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    input[static_cast<std::size_t>(i)] = samples[static_cast<std::size_t>(i)];
  }
  std::vector<std::complex<double>> spectrum(static_cast<std::size_t>(count));
  fft.forward(input, spectrum);
  const double scale = 2.0 * std::numbers::pi / period;
  for (int bin = 0; bin < count; ++bin) {
    int mode = bin <= count / 2 ? bin : bin - count;
    if (count % 2 == 0 && bin == count / 2) mode = 0;
    spectrum[static_cast<std::size_t>(bin)] *=
        std::complex<double>(0.0, scale * static_cast<double>(mode));
  }
  std::vector<std::complex<double>> back(static_cast<std::size_t>(count));
  fft.backward(spectrum, back);
  std::vector<double> derivative(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    derivative[static_cast<std::size_t>(i)] =
        back[static_cast<std::size_t>(i)].real() / static_cast<double>(count);
  }
  return derivative;
}

inline std::vector<double> jacobians(const PeriodicClusterMap &map, int count) {
  std::vector<double> weight(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    const double xi = cluster_computational(map, i, count);
    const double jacobian = cluster_jacobian(map, xi);
    if (!(jacobian > 0.0)) {
      throw std::invalid_argument("periodic cluster Jacobian is not positive");
    }
    weight[static_cast<std::size_t>(i)] = jacobian;
  }
  return weight;
}

} // namespace cluster_detail

[[nodiscard]] inline std::vector<double>
cluster_derivative(const PeriodicClusterMap &map, std::span<const double> samples) {
  const auto slope = cluster_detail::xi_derivative(samples, map.period);
  const auto weight =
      cluster_detail::jacobians(map, static_cast<int>(samples.size()));
  std::vector<double> physical(slope.size());
  for (std::size_t i = 0; i < slope.size(); ++i) physical[i] = slope[i] / weight[i];
  return physical;
}

[[nodiscard]] inline std::vector<double>
cluster_second_derivative(const PeriodicClusterMap &map,
                          std::span<const double> samples) {
  const auto first = cluster_derivative(map, samples);
  return cluster_derivative(map, first);
}

/// Physical integral `∫ u dx = (L/N) Σ u(ξ) x'(ξ)`.
[[nodiscard]] inline double
cluster_physical_integral(const PeriodicClusterMap &map,
                          std::span<const double> samples) {
  const int count = static_cast<int>(samples.size());
  if (count < 2) {
    throw std::invalid_argument(
        "periodic cluster integral needs at least two nodes");
  }
  const auto weight = cluster_detail::jacobians(map, count);
  double sum = 0.0;
  for (int i = 0; i < count; ++i)
    sum +=
        samples[static_cast<std::size_t>(i)] * weight[static_cast<std::size_t>(i)];
  return sum * map.period / static_cast<double>(count);
}

} // namespace pfc::geometry
