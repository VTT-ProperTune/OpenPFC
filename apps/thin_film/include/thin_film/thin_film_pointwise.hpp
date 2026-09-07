// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file thin_film_pointwise.hpp
 * @brief Device-capable nonlinear disjoining remainder of the lubrication film.
 *
 * @details
 * Constant-mobility lubrication
 * \f$\partial_t h=\nabla\cdot\bigl[M_0\nabla(-\gamma\nabla^2 h-\Pi(h))\bigr]\f$
 * splits as \f$L(k)\hat h+M_{\mathrm{nl}}(k)\widehat{n}\f$ with
 * \f$n=\Pi(h)-\Pi(h_0)-\Pi'(h_0)(h-h_0)\f$.
 *
 * Disjoining pressure (nondimensional, mean thickness \(h_0\)):
 * \f$\Pi(h)=A\bigl((h_0/h)^3-(h_0/h)^9\bigr)\f$ (van der Waals attraction plus
 * short-range repulsion). Height is clamped away from zero.
 */

#include <cmath>

#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/kernel/simulation/spectral_pointwise.hpp>

namespace thin_film {

struct ThinFilmPointwise {
  double A{0.05};   ///< disjoining strength
  double h0{1.0};   ///< linearization thickness
  double Pi0{0.0};  ///< \f$\Pi(h_0)\f$
  double Pip0{0.0}; ///< \f$\Pi'(h_0)\f$

  static constexpr double kMinH = 1.0e-4;

  [[nodiscard]] OPENPFC_HD double clamp_h(double h) const {
    return (h < kMinH) ? kMinH : h;
  }

  [[nodiscard]] OPENPFC_HD double Pi(double h) const {
    const double u = clamp_h(h) / h0;
    const double u3 = u * u * u;
    const double u9 = u3 * u3 * u3;
    return A * (1.0 / u3 - 1.0 / u9);
  }

  [[nodiscard]] OPENPFC_HD double Pi_prime(double h) const {
    const double hh = clamp_h(h);
    const double u = hh / h0;
    const double u4 = u * u * u * u;
    const double u10 = u4 * u4 * u * u;
    return A * (-3.0 / u4 + 9.0 / u10) / h0;
  }

  /// Potential with \f$V'=-\Pi\f$ so the gradient flow decreases \f$\int V\f$.
  [[nodiscard]] OPENPFC_HD double V(double h) const {
    const double u = clamp_h(h) / h0;
    const double u2 = u * u;
    const double u8 = u2 * u2 * u2 * u2;
    return A * h0 * (0.5 / u2 - 0.125 / u8);
  }

  [[nodiscard]] OPENPFC_HD double n_nl(double h) const {
    return Pi(h) - Pi0 - Pip0 * (h - h0);
  }

  [[nodiscard]] OPENPFC_HD double
  nonlinearity(const pfc::sim::SpectralCell &cell) const {
    return n_nl(cell.psi);
  }

  [[nodiscard]] OPENPFC_HD double
  free_energy_density(const pfc::sim::SpectralCell &cell) const {
    return V(cell.psi);
  }
};

} // namespace thin_film
