// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file cases.hpp
 * @brief Initial conditions and exact solutions for the 2-D NS prototype.
 *
 * Taylor–Green lives on \f$[0,2\pi]^2\f$. The Minion–Brown / Bell double
 * shear layer is the unit-square formulation; do not transplant
 * \f$\rho=30\f$ onto \f$[0,2\pi]^2\f$.
 */

#include <cmath>
#include <stdexcept>
#include <string_view>

#include <openpfc/kernel/data/constants.hpp>

namespace ns2d {

enum class Case { taylor_green, shear, two_mode };

[[nodiscard]] inline const char *case_name(Case c) noexcept {
  switch (c) {
  case Case::taylor_green: return "taylor_green";
  case Case::shear: return "shear";
  case Case::two_mode: return "two_mode";
  }
  return "unknown";
}

[[nodiscard]] inline Case parse_case(std::string_view s) {
  if (s == "taylor_green" || s == "tg") return Case::taylor_green;
  if (s == "shear" || s == "kh" || s == "kelvin_helmholtz") return Case::shear;
  if (s == "two_mode") return Case::two_mode;
  throw std::invalid_argument(
      "unknown ns2d case (expected taylor_green|shear|two_mode)");
}

/// Exact Taylor–Green streamfunction on \f$[0,2\pi]^2\f$:
/// \f$\psi=\sin x\sin y\,e^{-2\nu t}\f$.
[[nodiscard]] inline double taylor_green_psi(double x, double y, double nu,
                                             double t) noexcept {
  return std::sin(x) * std::sin(y) * std::exp(-2.0 * nu * t);
}

[[nodiscard]] inline double taylor_green_u(double x, double y, double nu,
                                           double t) noexcept {
  return std::sin(x) * std::cos(y) * std::exp(-2.0 * nu * t);
}

[[nodiscard]] inline double taylor_green_v(double x, double y, double nu,
                                           double t) noexcept {
  return -std::cos(x) * std::sin(y) * std::exp(-2.0 * nu * t);
}

[[nodiscard]] inline double taylor_green_omega(double x, double y, double nu,
                                               double t) noexcept {
  return 2.0 * std::sin(x) * std::sin(y) * std::exp(-2.0 * nu * t);
}

[[nodiscard]] inline double taylor_green_ke(double nu, double t) noexcept {
  return 0.25 * std::exp(-4.0 * nu * t);
}

[[nodiscard]] inline double taylor_green_enstrophy(double nu, double t) noexcept {
  return 0.5 * std::exp(-4.0 * nu * t);
}

/**
 * @brief Minion–Brown / Bell–Colella–Glaz double shear on the unit square.
 *
 * Standard parameters (Minion & Brown, JCP 1997; Bell, Colella & Glaz):
 * \f$\rho=30\f$, \f$\varepsilon=0.05\f$, \f$\mathrm{Re}=10^4\f$ so
 * \f$\nu=10^{-4}\f$ with \f$U\sim 1\f$ and box length 1.
 *
 * \f[
 *   u=\begin{cases}
 *     \tanh(\rho(y-1/4)) & y\le 1/2\\
 *     \tanh(\rho(3/4-y)) & y>1/2
 *   \end{cases},\qquad
 *   v=\varepsilon\sin(2\pi x).
 * \f]
 * Vorticity is \f$\omega=\partial_x v-\partial_y u\f$.
 *
 * Characteristic thickness is \f$1/\rho\f$. Cells per thickness:
 * \f$N/\rho\f$ on an \f$N\times N\f$ unit-square mesh (before 2/3
 * dealiasing). After Orszag 2/3 the effective count is
 * \f$(2N/3)/\rho\f$.
 *
 * To pose the *same* physical layer on \f$[0,2\pi]^2\f$ one would set
 * \f$\rho'= \rho/(2\pi)\f$ and \f$\nu'=2\pi\nu\f$ (length scale
 * \f$2\pi\f$). This prototype does **not** do that: shear always uses
 * the unit square. Putting \f$\rho=30\f$ on \f$[0,2\pi]^2\f$ makes the
 * layer \f$2\pi\f$ times thinner than Minion–Brown and is not that
 * benchmark.
 */
[[nodiscard]] inline double double_shear_omega(double x, double y, double rho,
                                               double eps) noexcept {
  const auto sech2 = [](double z) {
    const double c = std::cosh(z);
    return 1.0 / (c * c);
  };
  const double dudy =
      (y <= 0.5) ? rho * sech2(rho * (y - 0.25)) : -rho * sech2(rho * (0.75 - y));
  return 2.0 * pfc::pi * eps * std::cos(2.0 * pfc::pi * x) - dudy;
}

/// Two Fourier modes on \f$[0,2\pi]^2\f$: nonlinear IC for timestep tests.
[[nodiscard]] inline double two_mode_omega(double x, double y) noexcept {
  return std::sin(x) * std::sin(y) + 0.5 * std::sin(2.0 * x) * std::sin(2.0 * y);
}

[[nodiscard]] inline double shear_thickness(double rho) noexcept {
  return 1.0 / rho;
}

[[nodiscard]] inline double shear_cells_per_thickness(int n, double rho) noexcept {
  return static_cast<double>(n) / rho;
}

[[nodiscard]] inline double
shear_effective_cells_per_thickness(int n, double rho) noexcept {
  return (2.0 / 3.0) * static_cast<double>(n) / rho;
}

} // namespace ns2d
