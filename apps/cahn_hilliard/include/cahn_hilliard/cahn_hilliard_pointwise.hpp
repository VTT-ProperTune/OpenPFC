// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file cahn_hilliard_pointwise.hpp
 * @brief Device-capable bulk chemical-potential remainder of regular-solution CH.
 *
 * @details
 * Conserved Cahn–Hilliard with constant mobility splits as
 * \f$\partial_t \hat c = L(k)\hat c + M k_{\mathrm{lap}}\widehat{n}\f$
 * where \f$n(c)\f$ is the bulk \f$f'(c)\f$ with the linearization about the
 * alloy composition \f$c_0\f$ removed (so ETD treats \f$f''(c_0)\nabla^2 c\f$
 * and the \f$\kappa\nabla^4\f$ term exactly).
 *
 * Energy unit is \f$RT\f$. The bulk term is a Redlich-Kister substitutional
 * solution,
 *
 * \f[
 *   f(c)=c(1-c)\bigl[\omega+\ell_1(1-2c)\bigr]+c\ln c+(1-c)\ln(1-c),
 * \f]
 *
 * where \f$\omega=L_0/(RT)\f$ and \f$\ell_1=L_1/(RT)\f$. Setting
 * \f$\ell_1=0\f$ recovers the regular-solution model exactly, so the reduced
 * verifier is the degenerate case of this expression rather than a second code
 * path. See `fe_cr_thermo.hpp` for the coefficients and their provenance.
 * Composition is clamped away from \f$\{0,1\}\f$ so the logs stay finite.
 */

#include <cmath>

#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/kernel/simulation/spectral_pointwise.hpp>

namespace cahn_hilliard {

struct CahnHilliardPointwise {
  double omega_nd{3.23}; ///< \f$L_0/(RT)\f$ (regular-solution \f$\omega\f$)
  double l1_nd{0.0};     ///< \f$L_1/(RT)\f$; 0 gives the regular solution
  double c0{0.32};       ///< linearization composition (mole fraction Cr)
  double fprime0{0.0};   ///< \f$f'(c_0)\f$
  double fpp0{0.0};      ///< \f$f''(c_0)\f$

  static constexpr double kMinC = 1.0e-12;
  static constexpr double kMaxC = 1.0 - 1.0e-12;

  [[nodiscard]] OPENPFC_HD double clamp_c(double c) const {
    return (c < kMinC) ? kMinC : ((c > kMaxC) ? kMaxC : c);
  }

  [[nodiscard]] OPENPFC_HD double f_bulk(double c) const {
    const double u = clamp_c(c);
    return u * (1.0 - u) * (omega_nd + l1_nd * (1.0 - 2.0 * u)) +
           u * std::log(u) + (1.0 - u) * std::log(1.0 - u);
  }

  [[nodiscard]] OPENPFC_HD double f_prime(double c) const {
    const double u = clamp_c(c);
    return omega_nd * (1.0 - 2.0 * u) +
           l1_nd * (1.0 - 6.0 * u + 6.0 * u * u) + std::log(u / (1.0 - u));
  }

  [[nodiscard]] OPENPFC_HD double f_double_prime(double c) const {
    const double u = clamp_c(c);
    return -2.0 * omega_nd + l1_nd * (12.0 * u - 6.0) + 1.0 / (u * (1.0 - u));
  }

  /// Bulk \f$f'(c)\f$ minus the part already in \f$L(k)\f$.
  [[nodiscard]] OPENPFC_HD double n_nl(double c) const {
    return f_prime(c) - fprime0 - fpp0 * (c - c0);
  }

  [[nodiscard]] OPENPFC_HD double
  nonlinearity(const pfc::sim::SpectralCell &cell) const {
    return n_nl(cell.psi);
  }

  /// Bulk free-energy density (no \f$(\kappa/2)|\nabla c|^2\f$ term).
  [[nodiscard]] OPENPFC_HD double
  free_energy_density(const pfc::sim::SpectralCell &cell) const {
    return f_bulk(cell.psi);
  }
};

} // namespace cahn_hilliard
