// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

// Complete-domain teaching helpers, not a public curvilinear-grid API.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <numbers>
#include <openpfc/kernel/fft/local_fft.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace wave {
using Vector = std::vector<double>;

inline nlohmann::json input(int argc, char **argv) {
  if (argc < 2) return nlohmann::json::object();
  std::ifstream file(argv[1]);
  if (!file) throw std::runtime_error("Cannot open input JSON");
  nlohmann::json config;
  file >> config;
  return config;
}

inline std::ofstream output(const char *name) {
  std::filesystem::create_directories("results");
  std::ofstream file(std::string("results/") + name);
  if (!file) throw std::runtime_error("Cannot open output CSV");
  file.exceptions(std::ios::badbit | std::ios::failbit);
  file << std::setprecision(17);
  return file;
}

inline void positive(double value, const char *name) {
  if (!std::isfinite(value) || value <= 0)
    throw std::invalid_argument(std::string(name) + " must be finite and positive");
}

// Round the number of steps per output interval as in the book.
struct Schedule {
  int gap, frames;
  double dt;

  Schedule(double requested, double interval, double end) {
    positive(requested, "dt");
    positive(interval, "saveat");
    positive(end, "end");
    if (interval / requested > 1000000 || end / interval > 10000)
      throw std::invalid_argument("Too many steps or output frames");
    gap = std::max(1, static_cast<int>(std::round(interval / requested)));
    frames = std::max(1, static_cast<int>(std::round(end / interval)));
    dt = interval / gap;
    if (static_cast<double>(gap) * frames > 2000000)
      throw std::invalid_argument("Too many timesteps");
  }
};

class ChebyshevFFT {
  int n_;
  pfc::fft::LocalFFT fft_;
  Vector extended_, theta_first_, theta_second_;
  pfc::fft::ComplexVector coefficients_, first_, second_;

public:
  const Vector x;

  explicit ChebyshevFFT(int n)
      : n_(n), fft_({2 * n, 1, 1}), extended_(2 * n), theta_first_(2 * n),
        theta_second_(2 * n), coefficients_(n + 1), first_(n + 1), second_(n + 1),
        x([n] {
          Vector nodes(n + 1);
          for (int j = 0; j <= n; ++j) nodes[j] = std::cos(std::numbers::pi * j / n);
          return nodes;
        }()) {}

  // start snippet cosine
  Vector derivative(const Vector &u, int order) {
    if (u.size() != x.size() || (order != 1 && order != 2))
      throw std::invalid_argument("Use n+1 samples and derivative order 1 or 2");
    for (int j = 0; j < 2 * n_; ++j) extended_[j] = u[j <= n_ ? j : 2 * n_ - j];
    fft_.forward(extended_, coefficients_);
    for (int k = 0; k <= n_; ++k) {
      first_[k] = k == n_ ? 0.0 : std::complex<double>(0, k) * coefficients_[k];
      second_[k] = -static_cast<double>(k * k) * coefficients_[k];
    }
    fft_.backward(first_, theta_first_);
    if (order == 2) fft_.backward(second_, theta_second_);
    Vector result(n_ + 1);
    for (int j = 1; j < n_; ++j) {
      const double sine = std::sin(std::numbers::pi * j / n_);
      result[j] = order == 1 ? -theta_first_[j] / sine
                             : theta_second_[j] / (sine * sine) -
                                   x[j] * theta_first_[j] / (sine * sine * sine);
    }
    // Analytic limits of T'_k and T''_k avoid dividing by sin(theta)=0.
    for (int k = 1; k <= n_; ++k) {
      const double a = coefficients_[k].real() / (k == n_ ? 2 * n_ : n_);
      const double endpoint = order == 1 ? k * k : k * k * (k * k - 1.0) / 3;
      result[0] += a * endpoint;
      result[n_] += a * endpoint * ((k + order) % 2 ? -1 : 1);
    }
    return result;
  }
  // end snippet cosine
};
} // namespace wave
