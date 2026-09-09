// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file ehd_film_pointwise.hpp
 * @brief Nonlinear disjoining remainder of the EHD film.
 *
 * \(n=\Pi(h)-\Pi(h_0)-\Pi'(h_0)(h-h_0)\). Same van der Waals plus repulsion
 * as the lubrication app; copied rather than shared.
 *
 * `h_star` adds the same rupture/adhesion-safe switch `thin_film` uses
 * (`#114`): zero keeps the original two-term potential
 * \(\Pi=A\bigl((h_0/h)^3-(h_0/h)^9\bigr)\), destabilizing at \(h_0\) but
 * with no stable state as \(h\to0\); a positive value switches to the
 * precursor form \(\Pi=A\bigl[(h_{\mathrm{star}} / h)^9
 * -(h_{\mathrm{star}} / h)^3\bigr]\), repulsive below \(h_{\mathrm{star}}\)
 * and attractive above it, so a plate pinned down by adhesion and load
 * settles on a stable precursor gap instead of the pressure diverging.
 * Default is `0`, so the existing `#81` linear verifier is unchanged.
 */

#include <cmath>

#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/kernel/simulation/spectral_pointwise.hpp>

namespace ehd_film {

struct EhdFilmPointwise {
  double A{0.0};
  double h0{1.0};
  double h_star{0.0};
  double Pi0{0.0};
  double Pip0{0.0};

  static constexpr double kMinH = 1.0e-4;

  [[nodiscard]] OPENPFC_HD double clamp_h(double h) const {
    return (h < kMinH) ? kMinH : h;
  }

  [[nodiscard]] OPENPFC_HD double Pi(double h) const {
    if (h_star > 0.0) {
      const double v = h_star / clamp_h(h);
      const double v3 = v * v * v;
      return A * (v3 * v3 * v3 - v3);
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
      const double v3 = v * v * v;
      return A * (-9.0 * v3 * v3 * v3 + 3.0 * v3) / hh;
    }
    const double u = hh / h0;
    const double u4 = u * u * u * u;
    const double u10 = u4 * u4 * u * u;
    return A * (-3.0 / u4 + 9.0 / u10) / h0;
  }

  [[nodiscard]] OPENPFC_HD double n_nl(double h) const {
    return Pi(h) - Pi0 - Pip0 * (h - h0);
  }

  [[nodiscard]] OPENPFC_HD double
  nonlinearity(const pfc::sim::SpectralCell &cell) const {
    return n_nl(cell.psi);
  }
};

} // namespace ehd_film
