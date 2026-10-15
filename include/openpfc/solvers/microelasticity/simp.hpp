// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file simp.hpp
 * @brief SIMP stiffness interpolation of a design density.
 *
 * @details
 * The solid isotropic material with penalization (SIMP) law
 *
 * \f[
 *   \mathbf C(\phi) = \mathbf C_{\min} + \phi^{p}\,(\mathbf C_{\max}
 *                      - \mathbf C_{\min}),\qquad \phi\in[0,1],
 * \f]
 *
 * is the linear blend of `EigenstrainMicroelasticity` with stiffness weight
 * \f$w=\phi^{p}\f$, `stiffness_at_zero` \f$=\mathbf C_{\min}\f$ and
 * `stiffness_at_one` \f$=\mathbf C_{\max}\f$. A penalty \f$p>1\f$ makes
 * intermediate densities carry less stiffness than their volume costs, so an
 * optimizer that rewards stiffness drives the density towards 0 or 1
 * (Bendsøe & Sigmund, *Arch. Appl. Mech.* **69**, 635 (1999)). \f$p=1\f$ is
 * the plain linear blend and never evaluates \f$\mathrm{pow}(\phi,0)\f$.
 * \f$\phi\f$ is clamped to \f$[0,1]\f$ first, so \f$\phi=0\f$ gives no NaN
 * for \f$p\ge 1\f$.
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pfc::solvers {

/// \f$w(\phi)=\phi^{p}\f$ and \f$w'(\phi)=p\,\phi^{p-1}\f$, \f$\phi\f$
/// clamped to \f$[0,1]\f$.
struct SimpInterpolation {
  double exponent{3.0};

  [[nodiscard]] static double clamp01(double phi) noexcept {
    return std::min(1.0, std::max(0.0, phi));
  }

  [[nodiscard]] double weight(double phi) const noexcept {
    phi = clamp01(phi);
    if (exponent == 1.0) return phi;
    return std::pow(phi, exponent);
  }

  [[nodiscard]] double slope(double phi) const noexcept {
    phi = clamp01(phi);
    if (exponent == 1.0) return 1.0;
    return exponent * std::pow(phi, exponent - 1.0);
  }

  /// @throws std::invalid_argument unless \f$p\ge 1\f$ and finite.
  void validate() const {
    if (!(exponent >= 1.0) || !std::isfinite(exponent)) {
      throw std::invalid_argument("SimpInterpolation: exponent must be >= 1");
    }
  }
};

} // namespace pfc::solvers
