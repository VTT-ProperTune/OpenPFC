// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file capillary_gravity_mapping.hpp
 * @brief Physical/nondimensional capillary-gravity mapping to (alpha, beta, gamma) (`#119`).
 *
 * @details
 * The Kawahara equation is the standard long-wave reduction of the free-surface
 * water-wave problem near the **critical Bond number** \(\tau=1/3\), where the
 * third-order (KdV) dispersion coefficient becomes small and a fifth-order term
 * is needed to leading order (Hasimoto 1970; Kawahara 1972, J. Phys. Soc. Jpn.
 * 33, 260; Hunter & Vanden-Broeck 1983, J. Fluid Mech. 134, 205). For water of
 * undisturbed depth \(h\), gravity \(g\), surface tension \(T\) and density
 * \(\rho\), the Bond number is
 *
 * \f[
 *   \tau = \frac{T}{\rho g h^2},
 * \f]
 *
 * and in a frame translating at the linear long-wave speed \(c_0=\sqrt{gh}\)
 * (i.e. \(\xi=x-c_0t\), which removes the leading-order \(c_0\eta_x\)
 * advection and isolates the dispersive/nonlinear correction), the free-surface
 * elevation \(\eta(\xi,t)\) obeys, to the order retained in the reduction,
 *
 * \f[
 *   \eta_t + \frac{3c_0}{2h}\,\eta\eta_\xi
 *   + \frac{c_0h^2}{6}(1-3\tau)\,\eta_{\xi\xi\xi}
 *   + \frac{c_0h^4}{90}\,\eta_{\xi\xi\xi\xi\xi} = 0.
 * \f]
 *
 * Matched against this app's convention
 * \(u_t+\alpha uu_x-\beta u_{xxx}+\gamma u_{xxxxx}=0\) (`kawahara_physics.hpp`),
 *
 * \f[
 *   \alpha = \frac{3c_0}{2h}, \qquad
 *   \beta  = \frac{c_0h^2}{6}(3\tau-1), \qquad
 *   \gamma = \frac{c_0h^4}{90}.
 * \f]
 *
 * \f$\gamma>0\f$ always. \f$\beta<0\f$ (competing dispersion, real crossover
 * \(k_c=\sqrt{-\beta/\gamma}\)) for \(\tau<1/3\); \(\beta=0\) exactly at the
 * critical Bond number; \(\beta>0\) (no real crossover, monotone \(c_p\)) for
 * \(\tau>1/3\). This is the same crossover the app already documents for the
 * arbitrary-coefficient verification tests, now with a physically named
 * control parameter (\(\tau\)) instead of hand-picked numbers.
 *
 * ## Asymptotic validity
 *
 * The reduction assumes (a) long waves, \(kh\ll1\) relative to the leading
 * KdV balance, with the fifth-order term retained only because the standard
 * third-order coefficient is small near \(\tau=1/3\); (b) weak nonlinearity,
 * wave steepness \(A/h\ll1\), balanced against the (small) dispersion so the
 * KdV/Kawahara scaling remains self-consistent; (c) irrotational, inviscid,
 * single-layer flow over a flat bottom; (d) one horizontal dimension. None of
 * these are enforced numerically by this app -- `alpha`/`beta`/`gamma` are
 * free JSON parameters and the ETD integrator will happily run outside the
 * regime where the asymptotics hold.
 *
 * ## Honesty / provenance note (required by `#119`)
 *
 * The coefficient formula above was assembled from secondary/tertiary
 * literature summaries (review articles and their equation excerpts,
 * retrieved by web search on 2026-09-09) that consistently attribute it to
 * Hasimoto (1970) and Kawahara (1972), with the near-critical-Bond-number
 * analysis credited to Hunter & Vanden-Broeck (1983). This machine has no
 * general internet access at build/run time, and the primary Kawahara (1972)
 * and Hasimoto (1970) papers were not directly readable during this work (old
 * journal issues, not open-access, not mirrored in this repository). The
 * three numeric prefactors (\(3/2\), \(1/6\), \(1/90\)) and the \((1-3\tau)\)
 * structure recur consistently across the independent sources found, which is
 * why they are used here, but they have **not** been independently re-derived
 * from the Euler water-wave equations nor cross-checked against a primary
 * source in this repository. Treat this mapping as **representative** of the
 * documented capillary-gravity regime, not as a **quantitative**,
 * primary-source-verified calibration. Do not cite this header as a
 * literature reference; cite Kawahara (1972) / Hunter & Vanden-Broeck (1983)
 * directly and re-derive or re-check the prefactors first if quantitative
 * accuracy is required.
 */

#include <cmath>

namespace kawahara {

/// Nondimensional (or physical, in a consistent unit system) capillary-gravity
/// input: depth `h`, gravity `g`, and Bond number `tau = T/(rho g h^2)`.
struct CapillaryGravityRegime {
  double h{1.0};   ///< undisturbed depth
  double g{1.0};   ///< gravitational acceleration
  double tau{0.3}; ///< Bond number T/(rho g h^2); critical value is 1/3
};

/// Linear long-wave speed c0 = sqrt(g h).
[[nodiscard]] inline double c0_of(const CapillaryGravityRegime &r) {
  return std::sqrt(r.g * r.h);
}

/// Nonlinear coefficient alpha = 3 c0 / (2 h) (app convention: +alpha*u*u_x).
[[nodiscard]] inline double alpha_of(const CapillaryGravityRegime &r) {
  return 1.5 * c0_of(r) / r.h;
}

/// Third-order coefficient beta = (c0 h^2/6)(3 tau - 1)
/// (app convention: -beta*u_xxx, so a documented "+b*u_xxx" term is b=-beta).
[[nodiscard]] inline double beta_of(const CapillaryGravityRegime &r) {
  const double c0 = c0_of(r);
  return (c0 * r.h * r.h / 6.0) * (3.0 * r.tau - 1.0);
}

/// Fifth-order coefficient gamma = c0 h^4 / 90 (app convention: +gamma*u_xxxxx).
[[nodiscard]] inline double gamma_of(const CapillaryGravityRegime &r) {
  const double c0 = c0_of(r);
  const double h2 = r.h * r.h;
  return c0 * h2 * h2 / 90.0;
}

/// Competing-dispersion crossover k_c = sqrt(-beta/gamma) (phase velocity
/// c_p = beta*k^2 + gamma*k^4 changes sign here). Only meaningful/real when
/// beta*gamma < 0, i.e. tau < 1/3 for this mapping (gamma > 0 always).
[[nodiscard]] inline double crossover_k(const CapillaryGravityRegime &r) {
  const double b = beta_of(r);
  const double c = gamma_of(r);
  return std::sqrt(-b / c);
}

} // namespace kawahara
