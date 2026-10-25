// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file mhd_cases.hpp
 * @brief Initial conditions for 2-D incompressible MHD (#23, #113).
 *
 * Orszag–Tang is the incompressible periodic vortex of Orszag & Tang,
 * JFM 90 (1979), not a compressible shock-tube variant.
 *
 * Island coalescence is the Ng & Ragunathan (arXiv:1106.0521) flux
 * on OpenPFC's [0,2π]² domain, with a frozen deterministic
 * streamfunction seed. The amplitude is not retuned with η.
 */

#include <array>
#include <cmath>
#include <stdexcept>
#include <string_view>

#include <openpfc/kernel/data/constants.hpp>

namespace ns2d {

enum class MHDCase {
  orszag_tang,
  hydro_control,
  force_free,
  alfven,
  island_coalescence
};

[[nodiscard]] inline const char *mhd_case_name(MHDCase c) noexcept {
  switch (c) {
  case MHDCase::orszag_tang: return "orszag_tang";
  case MHDCase::hydro_control: return "hydro_control";
  case MHDCase::force_free: return "force_free";
  case MHDCase::alfven: return "alfven";
  case MHDCase::island_coalescence: return "island_coalescence";
  }
  return "unknown";
}

[[nodiscard]] inline MHDCase parse_mhd_case(std::string_view s) {
  if (s == "orszag_tang" || s == "ot") return MHDCase::orszag_tang;
  if (s == "hydro_control" || s == "hydro") return MHDCase::hydro_control;
  if (s == "force_free" || s == "diffuse") return MHDCase::force_free;
  if (s == "alfven") return MHDCase::alfven;
  if (s == "island_coalescence" || s == "coalescence" || s == "ic")
    return MHDCase::island_coalescence;
  throw std::invalid_argument(
      "unknown mhd case (expected orszag_tang|hydro_control|force_free|"
      "alfven|island_coalescence)");
}

/// φ = cos x + cos y  ⇒  u = (-sin y, sin x), ω = cos x + cos y.
[[nodiscard]] inline double ot_omega(double x, double y) noexcept {
  return std::cos(x) + std::cos(y);
}

/// a = (1/2) cos 2x + cos y  ⇒  B = (-sin y, sin 2x).
[[nodiscard]] inline double ot_a(double x, double y) noexcept {
  return 0.5 * std::cos(2.0 * x) + std::cos(y);
}

[[nodiscard]] inline double force_free_a(double x, double y) noexcept {
  return std::sin(x) * std::sin(y);
}

[[nodiscard]] inline double force_free_a_exact(double x, double y, double eta,
                                               double t) noexcept {
  return force_free_a(x, y) * std::exp(-2.0 * eta * t);
}

/// Two-mode streamfunction shared by ω and a for the Alfvénic state u=B.
[[nodiscard]] inline double alfven_phi(double x, double y) noexcept {
  return std::sin(x) * std::sin(y) + 0.25 * std::sin(2.0 * x) * std::sin(2.0 * y);
}

[[nodiscard]] inline double alfven_omega(double x, double y) noexcept {
  return 2.0 * std::sin(x) * std::sin(y) +
         2.0 * std::sin(2.0 * x) * std::sin(2.0 * y);
}

/// Ng & Ragunathan \f$\bar A=0.4\f$ on the unit square, mapped to
/// \f$[0,2\pi]^2\f$ as \f$a=0.4\sin x\sin y\f$.
inline constexpr double coalescence_abar = 0.4;

/// Frozen Stage-0 streamfunction amplitude. Do not retune with η.
inline constexpr double coalescence_eps = 0.01;

/// \f$a=0.4\sin x\sin y\f$. Then \f$j=2a\f$ and \f$\mathbf B\cdot\nabla j=0\f$,
/// so the unperturbed field is an exact static magnetic state.
[[nodiscard]] inline double coalescence_a(double x, double y) noexcept {
  return coalescence_abar * std::sin(x) * std::sin(y);
}

/// Unperturbed coalescence flux decays as the \f$k^2=2\f$ eigenmode.
[[nodiscard]] inline double coalescence_a_exact(double x, double y, double eta,
                                                double t) noexcept {
  return coalescence_a(x, y) * std::exp(-2.0 * eta * t);
}

/// \f$\phi=\varepsilon(\cos x-\cos y)\f$ with frozen \f$\varepsilon=0.01\f$.
/// Then \f$\mathbf u=(\varepsilon\sin y,\varepsilon\sin x)\f$ and
/// \f$\omega=\varepsilon(\cos x-\cos y)\f$. This is the lowest-k
/// divergence-free strain that drives like-signed islands together
/// along the diagonals. See COALESCENCE.md.
[[nodiscard]] inline double coalescence_phi(double x, double y) noexcept {
  return coalescence_eps * (std::cos(x) - std::cos(y));
}

[[nodiscard]] inline double coalescence_omega(double x, double y) noexcept {
  return coalescence_eps * (std::cos(x) - std::cos(y));
}

[[nodiscard]] inline std::array<double, 2> coalescence_u(double x,
                                                         double y) noexcept {
  return {coalescence_eps * std::sin(y), coalescence_eps * std::sin(x)};
}

} // namespace ns2d
