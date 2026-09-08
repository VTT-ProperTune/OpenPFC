// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file higher_order_pfc_pointwise.hpp
 * @brief Device-capable local nonlinearity of the higher-order PFC free energy.
 *
 * @details
 * The whole quadratic part of the free energy sits in the spectral kernel, so
 * the only real-space work left is the local part of \f$\delta F/\delta\psi\f$:
 *
 * \f[
 *   f_{\mathrm{loc}}(\psi) = -\frac{g}{3}\psi^3 + \frac{1}{4}\psi^4,
 *   \qquad
 *   n(\psi) = \frac{\partial f_{\mathrm{loc}}}{\partial \psi}
 *           = \psi^3 - g\,\psi^2 .
 * \f]
 *
 * \f$g=0\f$ is the symmetric quartic PFC. A positive \f$g\f$ breaks the
 * \f$\psi\to-\psi\f$ symmetry, which is what stabilises triangular (2D) and
 * BCC (3D) phases at small quench depth; a nonzero mean density does the same
 * job for conserved dynamics.
 */

#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/kernel/simulation/spectral_pointwise.hpp>

namespace higher_order_pfc {

struct HigherOrderPFCPointwise {
  double g{0.0}; ///< cubic coefficient; 0 keeps \f$\psi\to-\psi\f$ symmetry

  /// \f$n(\psi)=\psi^3-g\psi^2\f$.
  [[nodiscard]] OPENPFC_HD double n_nl(double psi) const {
    return psi * psi * (psi - g);
  }

  [[nodiscard]] OPENPFC_HD double
  nonlinearity(const pfc::sim::SpectralCell &cell) const {
    return n_nl(cell.psi);
  }

  /// Local part of the free-energy density (the kernel part is spectral).
  [[nodiscard]] OPENPFC_HD double
  free_energy_density(const pfc::sim::SpectralCell &cell) const {
    const double p = cell.psi;
    const double p2 = p * p;
    return 0.25 * p2 * p2 - (g / 3.0) * p2 * p;
  }
};

} // namespace higher_order_pfc
