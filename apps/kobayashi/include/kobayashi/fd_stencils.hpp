// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file fd_stencils.hpp
 * @brief Per-cell Kobayashi explicit-Euler stages on a padded host Field.
 *
 * Neighbours are storage-halo offsets (`i±1`, `j±1`). Callers exchange
 * state (φ, T) before stage A and aux (ε, ε′, φx, φy) before stage B.
 */

#include <cmath>
#include <numbers>

#include <kobayashi/defaults.hpp>
#include <openpfc/kernel/data/grid_field.hpp>

namespace kobayashi {

using HostField = pfc::data::Field<double, pfc::HostSpace>;

inline void stage_a_cell(HostField &phi, HostField &tempr, HostField &lap_phi,
                         HostField &lap_t, HostField &phidx, HostField &phidy,
                         HostField &epsilon, HostField &epsilon_deriv, int i, int j,
                         int k, double inv_dx, double inv_dy, double inv_lap_den) {
  const double hne = phi(i + 1, j, k);
  const double hnw = phi(i - 1, j, k);
  const double hns = phi(i, j - 1, k);
  const double hnn = phi(i, j + 1, k);
  const double hnc = phi(i, j, k);
  lap_phi(i, j, k) = (hne + hnw + hns + hnn - 4.0 * hnc) * inv_lap_den;

  const double Tne = tempr(i + 1, j, k);
  const double Tnw = tempr(i - 1, j, k);
  const double Tns = tempr(i, j - 1, k);
  const double Tnn = tempr(i, j + 1, k);
  const double Tnc = tempr(i, j, k);
  lap_t(i, j, k) = (Tne + Tnw + Tns + Tnn - 4.0 * Tnc) * inv_lap_den;

  const double dpx = (phi(i + 1, j, k) - phi(i - 1, j, k)) * inv_dx;
  const double dpy = (phi(i, j + 1, k) - phi(i, j - 1, k)) * inv_dy;
  phidx(i, j, k) = dpx;
  phidy(i, j, k) = dpy;

  const double theta = std::atan2(dpy, dpx);
  epsilon(i, j, k) =
      kEpsilonb * (1.0 + kDelta * std::cos(kAniso * (theta - kTheta0)));
  epsilon_deriv(i, j, k) =
      -kEpsilonb * kAniso * kDelta * std::sin(kAniso * (theta - kTheta0));
}

inline void stage_b_cell(HostField &phi, HostField &tempr, HostField &lap_phi,
                         HostField &lap_t, HostField &phidx, HostField &phidy,
                         HostField &epsilon, HostField &epsilon_deriv, int i, int j,
                         int k, double inv_dx, double inv_dy, double dt) {
  const double phiold = phi(i, j, k);

  const double term1 =
      (epsilon(i, j + 1, k) * epsilon_deriv(i, j + 1, k) * phidx(i, j + 1, k) -
       epsilon(i, j - 1, k) * epsilon_deriv(i, j - 1, k) * phidx(i, j - 1, k)) *
      inv_dy;

  const double term2 =
      -(epsilon(i + 1, j, k) * epsilon_deriv(i + 1, j, k) * phidy(i + 1, j, k) -
        epsilon(i - 1, j, k) * epsilon_deriv(i - 1, j, k) * phidy(i - 1, j, k)) *
      inv_dx;

  const double ep = epsilon(i, j, k);
  const double term3 = ep * ep * lap_phi(i, j, k);

  const double m =
      kAlpha / std::numbers::pi * std::atan(kGamma * (kTeq - tempr(i, j, k)));
  const double term4 = phiold * (1.0 - phiold) * (phiold - 0.5 + m);

  phi(i, j, k) = phiold + (dt / kTau) * (term1 + term2 + term3 + term4);
  tempr(i, j, k) =
      tempr(i, j, k) + dt * lap_t(i, j, k) + kKappa * (phi(i, j, k) - phiold);
}

} // namespace kobayashi
