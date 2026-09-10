// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file resolution.hpp
 * @brief When a PFC grid is fine enough to dealias without losing physics.
 *
 * @details
 * PFC's nonlinearity is cubic, \f$\bar p_3\psi^2+\bar p_4\psi^3\f$, and the
 * density is periodic at the lattice wavenumber \f$k_0\f$ — which for this
 * app is \f$k_0=1\f$ by construction, since the correlation kernel is a
 * function of \f$|k|-1\f$. A crystal is therefore not a single mode: it
 * carries genuine content at \f$2k_0\f$ and \f$3k_0\f$, and the cubic term
 * generates more.
 *
 * Two independent conditions decide whether a grid can represent that
 * honestly, and both are about the same quantity \f$k_{\mathrm{Ny}}=\pi/\Delta
 * x\f$:
 *
 * 1. **The third harmonic must fit under Nyquist**, \f$3k_0\le
 *    k_{\mathrm{Ny}}\f$. Otherwise it folds back onto a resolved mode and
 *    deposits energy at a wavenumber the physics never put there.
 * 2. **The 2/3 dealiasing cut must sit above the second harmonic**,
 *    \f$2k_0\le\frac{2}{3}k_{\mathrm{Ny}}\f$. Otherwise switching the mask on
 *    deletes part of the crystal itself, which is a different error, not a
 *    fix for the first one.
 *
 * Both reduce to the same requirement:
 *
 * \f[
 *   \Delta x \;\le\; \frac{\pi}{3k_0},
 *   \qquad\text{i.e. at least } 6 \text{ grid points per lattice period.}
 * \f]
 *
 * Below that threshold a grid is squeezed from both sides: aliasing if the
 * mask is off, amputation if it is on. The shipped tungsten presets use
 * \f$\Delta x = 1.1107\f$, which is 6.1% coarser than \f$\pi/3\f$ — close
 * enough to have never looked wrong, far enough to be measurable. See
 * `tungsten_dealias_study` and the resolution section of the tungsten
 * chapter for what that costs.
 *
 * Exact dealiasing of a *cubic* term needs the 1/2 rule rather than the 2/3
 * rule, \f$2k_0\le\frac{1}{2}k_{\mathrm{Ny}}\f$, i.e. 8 points per period.
 * `dealias_clean_dx` returns that stricter figure; it is where the measured
 * mask-on/mask-off difference falls to the 1e-4 level.
 */

#include <numbers>

namespace tungsten::resolution {

/// The PFC lattice wavenumber this app's correlation kernel is built around.
inline constexpr double kLatticeWavenumber = 1.0;

/// \f$k_{\mathrm{Ny}}=\pi/\Delta x\f$.
[[nodiscard]] constexpr double nyquist_k(double dx) {
  return std::numbers::pi / dx;
}

/// Orszag 2/3 cutoff for this spacing.
[[nodiscard]] constexpr double two_thirds_cut(double dx) {
  return (2.0 / 3.0) * nyquist_k(dx);
}

/// Grid points per lattice period.
[[nodiscard]] constexpr double
points_per_lattice(double dx, double k0 = kLatticeWavenumber) {
  return 2.0 * std::numbers::pi / (k0 * dx);
}

/// Is the crystal's third harmonic representable, i.e. does nothing fold back?
[[nodiscard]] constexpr bool third_harmonic_resolved(double dx,
                                                     double k0 = kLatticeWavenumber) {
  return 3.0 * k0 <= nyquist_k(dx);
}

/// Would the 2/3 mask spare the crystal's own second harmonic?
[[nodiscard]] constexpr bool mask_spares_second_harmonic(
    double dx, double k0 = kLatticeWavenumber) {
  return 2.0 * k0 <= two_thirds_cut(dx);
}

/// Coarsest spacing at which both conditions hold: \f$\pi/(3k_0)\f$.
[[nodiscard]] constexpr double dealias_safe_dx(double k0 = kLatticeWavenumber) {
  return std::numbers::pi / (3.0 * k0);
}

/// Coarsest spacing at which the *cubic* term is exactly dealiased by the
/// 1/2 rule: \f$\pi/(4k_0)\f$, i.e. 8 points per lattice period.
[[nodiscard]] constexpr double dealias_clean_dx(double k0 = kLatticeWavenumber) {
  return std::numbers::pi / (4.0 * k0);
}

} // namespace tungsten::resolution
