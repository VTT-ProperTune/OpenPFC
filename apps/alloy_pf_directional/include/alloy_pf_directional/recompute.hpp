// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file recompute.hpp
 * @brief On-the-fly auxiliaries; default path still persists e^u and u.
 *
 * Persistent state (OpenMP / MPI / HIP):
 *   - φ (and ψ if Glasner) per grain
 *   - c
 *   - ∂tφ per grain (needed after the Euler update for antitrapping)
 *   - dc Jacobi scratch (solute RHS; not a physics field — keeps c frozen
 *     while the Ji stencil reads neighbor u)
 *   - e^u, u (default: stored; OPENPFC_ALCU_STORE_EU=0 recomputes)
 *
 * Recomputed from neighbors when a stencil needs them:
 *   - cubic anisotropy face fluxes (unless OPENPFC_ALCU_STORE_AUX=1)
 *   - ∇φ, |∇φ|² (Ji operators read φ/ψ directly)
 * Iso solute materializes nodal α (a_diff, a_at) and β once per cell
 * (xy-padded; 2D skips unused z-ghost planes) so div_alpha_grad is array
 * loads, not physics lambdas.
 *
 * Classic AMR is not implemented; see defaults.hpp and the app README.
 */

#include <alloy_pf_directional/defaults.hpp>
#include <cmath>

namespace alloy_pf_directional {

inline double eu_from_phi_c(double phi, double c, double ke, double clo) noexcept {
  double ci = c;
  if (std::isfinite(ci) && ci < kCMin && ci > -kFieldBlowAbs) {
    ci = kCMin;
  }
  const double den = std::max(denom_c(phi, ke), 1.0e-12);
  return (2.0 * ci / clo) / den;
}

inline double u_from_eu(double euv) noexcept { return std::log(std::max(euv, 1.0e-30)); }

/** Glasner antitrapping β = √2 atanh(φ). Hoist out of Ji neighbor visits. */
inline double beta_glasner_from_phi(double ph) noexcept {
  const double p = std::max(-0.999999, std::min(0.999999, ph));
  return std::sqrt(2.0) * std::atanh(p);
}

inline double a_at_nodal(double ph, double euv, double dte, double a_at, double A_trap,
                         double W0, double clo, double ke, bool use_glasner) noexcept {
  const double pref = a_prime_trap(ph, a_at, A_trap) * W0 * clo * (1.0 - ke) * euv * dte;
  if (use_glasner) {
    return pref * W0;
  }
  const double om = std::max(1.0 - ph * ph, 1.0e-8);
  return pref * std::sqrt(2.0) * W0 / om;
}

template <class F>
inline double phi_at(const F &p1, const F &p2, int n_grains, int i, int j, int k) noexcept {
  return (n_grains == 1) ? p1(i, j, k) : phi_eff_two(p1(i, j, k), p2(i, j, k));
}

template <class F>
inline double eu_at(const F &p1, const F &p2, const F &c, int n_grains, double ke, double clo,
                    int i, int j, int k) noexcept {
  return eu_from_phi_c(phi_at(p1, p2, n_grains, i, j, k), c(i, j, k), ke, clo);
}

/** Accessor for Ji `div(α ∇u)` so `u` is never stored. */
template <class F>
struct UFromC {
  const F &p1;
  const F &p2;
  const F &c;
  int n_grains = 1;
  double ke = kKe;
  double clo = kClo;
  double operator()(int i, int j, int k) const noexcept {
    return u_from_eu(eu_at(p1, p2, c, n_grains, ke, clo, i, j, k));
  }
};

inline double aniso_jx_face(double gx, double gy, double gz, const Physics &phys,
                            const Mat3 &R) noexcept {
  double jxv = 0.0, jyv = 0.0, jzv = 0.0, tau = 0.0, A = 0.0;
  cubic_aniso_from_grad(gx, gy, gz, phys.eps_c, phys.W0, phys.tau0, R, jxv, jyv, jzv, tau, A);
  (void)jyv;
  (void)jzv;
  (void)tau;
  (void)A;
  return jxv;
}
inline double aniso_jy_face(double gx, double gy, double gz, const Physics &phys,
                            const Mat3 &R) noexcept {
  double jxv = 0.0, jyv = 0.0, jzv = 0.0, tau = 0.0, A = 0.0;
  cubic_aniso_from_grad(gx, gy, gz, phys.eps_c, phys.W0, phys.tau0, R, jxv, jyv, jzv, tau, A);
  (void)jxv;
  (void)jzv;
  (void)tau;
  (void)A;
  return jyv;
}
inline double aniso_jz_face(double gx, double gy, double gz, const Physics &phys,
                            const Mat3 &R) noexcept {
  double jxv = 0.0, jyv = 0.0, jzv = 0.0, tau = 0.0, A = 0.0;
  cubic_aniso_from_grad(gx, gy, gz, phys.eps_c, phys.W0, phys.tau0, R, jxv, jyv, jzv, tau, A);
  (void)jxv;
  (void)jyv;
  (void)tau;
  (void)A;
  return jzv;
}

/** Face i+½. No-flux is encoded in ghosts (copy), not by local index. */
template <class F>
inline double flux_aniso_x(const F &pf, int i, int j, int k, int /*Nx*/, double inv_dx, bool dim3,
                           const Physics &phys, const Mat3 &R) noexcept {
  const double gx = inv_dx * (pf(i + 1, j, k) - pf(i, j, k));
  const double gy =
      0.25 * inv_dx * (pf(i + 1, j + 1, k) + pf(i, j + 1, k) - pf(i + 1, j - 1, k) - pf(i, j - 1, k));
  const double gz =
      dim3 ? 0.25 * inv_dx *
                 (pf(i + 1, j, k + 1) + pf(i, j, k + 1) - pf(i + 1, j, k - 1) - pf(i, j, k - 1))
           : 0.0;
  return aniso_jx_face(gx, gy, gz, phys, R);
}

template <class F>
inline double flux_aniso_y(const F &pf, int i, int j, int k, int /*Ny*/, double inv_dx, bool dim3,
                           bool /*periodic_y*/, const Physics &phys, const Mat3 &R) noexcept {
  const double gy = inv_dx * (pf(i, j + 1, k) - pf(i, j, k));
  const double gx =
      0.25 * inv_dx * (pf(i + 1, j + 1, k) + pf(i + 1, j, k) - pf(i - 1, j + 1, k) - pf(i - 1, j, k));
  const double gz =
      dim3 ? 0.25 * inv_dx *
                 (pf(i, j + 1, k + 1) + pf(i, j, k + 1) - pf(i, j + 1, k - 1) - pf(i, j, k - 1))
           : 0.0;
  return aniso_jy_face(gx, gy, gz, phys, R);
}

template <class F>
inline double flux_aniso_z(const F &pf, int i, int j, int k, int /*Nz*/, double inv_dx,
                           bool /*periodic_z*/, const Physics &phys, const Mat3 &R) noexcept {
  const double gz = inv_dx * (pf(i, j, k + 1) - pf(i, j, k));
  const double gx =
      0.25 * inv_dx * (pf(i + 1, j, k + 1) + pf(i + 1, j, k) - pf(i - 1, j, k + 1) - pf(i - 1, j, k));
  const double gy =
      0.25 * inv_dx * (pf(i, j + 1, k + 1) + pf(i, j + 1, k) - pf(i, j - 1, k + 1) - pf(i, j - 1, k));
  return aniso_jz_face(gx, gy, gz, phys, R);
}

} // namespace alloy_pf_directional
