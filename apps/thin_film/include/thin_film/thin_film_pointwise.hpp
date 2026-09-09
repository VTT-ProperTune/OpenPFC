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
  /**
   * Precursor thickness. Zero keeps the original two-term potential, which is
   * unstable at \f$h_0\f$ but has **no stable state at small \f$h\f$**: once
   * a hole opens, \f$(h_0/h)^9\f$ runs away and the solution diverges. That is
   * fine for the linear verifier, which never leaves the small-amplitude band,
   * and fatal for a rupture study.
   *
   * A positive value switches to the precursor form
   * \f$\Pi = A[(h_*\!/h)^3 - (h_*\!/h)^2]\f$, which is repulsive below
   * \f$h_*\f$ and attractive above it: the film still destabilises at
   * \f$h_0\f$ (\f$\Pi'(h_0) = A(2h_*^2 - 3h_*^3)/h_0^3 > 0\f$ for
   * \f$h_*<2h_0/3\f$) but a rupturing hole drains to a stable precursor
   * rather than to zero.
   *
   * The exponents are 3 and 2 rather than the 9 and 3 of the bulk form, and
   * that choice is numerical as much as physical. A ninth-power repulsion
   * calibrated to the same \f$\Pi'(h_0)\f$ reaches \f$\Pi\sim10^{6}\f$ by
   * \f$h=0.05h_0\f$, so a single cell dipping into the precursor produces a
   * NaN on the next step regardless of the timestep. The cubic pair is some
   * four thousand times softer there and integrates stably through rupture.
   */
  double h_star{0.0};
  double Pi0{0.0};  ///< \f$\Pi(h_0)\f$
  double Pip0{0.0}; ///< \f$\Pi'(h_0)\f$

  static constexpr double kMinH = 1.0e-4;

  /**
   * @brief Floor on the thickness used by the potential.
   *
   * Without a precursor the only job is to keep the logarithm-free powers
   * finite, and `kMinH` does that. With a precursor the floor matters
   * physically: a cell that overshoots to \f$h\le 0\f$ would otherwise give
   * \f$(h_*\!/h)^3\sim10^{10}\f$ and NaN the next step. Bounding the floor
   * at a fraction of \f$h_*\f$ keeps the repulsion large enough to push the
   * cell back but finite enough to integrate.
   *
   * Reaching this floor means the run is under-resolved, not converged; the
   * driver reports the minimum thickness so it can be checked.
   */
  [[nodiscard]] OPENPFC_HD double clamp_h(double h) const {
    const double floor = (h_star > 0.0) ? 0.25 * h_star : kMinH;
    return (h < floor) ? floor : h;
  }

  [[nodiscard]] OPENPFC_HD double Pi(double h) const {
    if (h_star > 0.0) {
      const double v = h_star / clamp_h(h);
      return A * (v * v * v - v * v);
    }
    const double u = clamp_h(h) / h0;
    const double u3 = u * u * u;
    const double u9 = u3 * u3 * u3;
    return A * (1.0 / u3 - 1.0 / u9);
  }

  [[nodiscard]] OPENPFC_HD double Pi_prime(double h) const {
    const double hh = clamp_h(h);
    if (h_star > 0.0) {
      const double v = h_star / hh;
      // d/dh [ (h*/h)^3 - (h*/h)^2 ] = ( -3 (h*/h)^3 + 2 (h*/h)^2 ) / h
      return A * (-3.0 * v * v * v + 2.0 * v * v) / hh;
    }
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
