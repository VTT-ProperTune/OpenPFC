// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <cmath>
#include <cstdint>

namespace alloy_pf_karma2001_benchmark {

inline std::uint64_t splitmix64(std::uint64_t &x) noexcept {
  x += 0x9E3779B97F4A7C15ULL;
  std::uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

/** Unit Gaussian from (seed, step, i, j, k, grain). OpenMP-safe, no shared RNG. */
inline double gaussian_n01(unsigned seed, int step, int i, int j, int k, int grain) noexcept {
  std::uint64_t x = static_cast<std::uint64_t>(seed) + 1ULL;
  x ^= 0xA24BAED4963EE407ULL * static_cast<std::uint64_t>(step + 1);
  x ^= 0x9E3779B97F4A7C15ULL * (static_cast<std::uint64_t>(static_cast<unsigned>(i)) + 1ULL);
  x ^= 0xBF58476D1CE4E5B9ULL * (static_cast<std::uint64_t>(static_cast<unsigned>(j)) + 1ULL);
  x ^= 0x94D049BB133111EBULL * static_cast<std::uint64_t>(grain + 1);
  if (k != 0) {
    x ^= 0x2545F4914F6CDD1DULL * (static_cast<std::uint64_t>(static_cast<unsigned>(k)) + 1ULL);
  }
  const std::uint64_t a = splitmix64(x);
  const std::uint64_t b = splitmix64(x);
  constexpr double scale = 1.0 / static_cast<double>(UINT64_C(1) << 53);
  const double u1 = (static_cast<double>(a >> 11) * scale < 1.0e-16)
                        ? 1.0e-16
                        : static_cast<double>(a >> 11) * scale;
  const double u2 = static_cast<double>(b >> 11) * scale;
  constexpr double twopi = 6.283185307179586476925286766559;
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(twopi * u2);
}

/** Rate added to ∂tφ: Euler–Maruyama for ⟨ηη⟩ = 2 (F/τ) g² δ(x)δ(t), g=1−φ². */
inline double fdt_phi_noise_rate(double F0, double W0, int n_dim, double tau, double dt, double dV,
                                 double phi, double xi) noexcept {
  if (!(F0 > 0.0) || !(tau > 0.0) || !(dt > 0.0) || !(dV > 0.0)) {
    return 0.0;
  }
  const double g = 1.0 - phi * phi;
  if (g < 1.0e-6) {
    return 0.0;
  }
  const double F = F0 * std::pow(W0, n_dim);
  return std::sqrt(2.0 * F / (tau * dt * dV)) * g * xi;
}

} // namespace alloy_pf_karma2001_benchmark
