// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file parameters.hpp
 * @brief Model parameters and thin-interface relations for the quantitative
 *        dilute-alloy phase field, equations (1)-(4) of `MODEL_SPEC.md`.
 *
 * @details
 * ### Why this header exists separately from the stepper
 *
 * Every number a Stage-1 verification compares against is a *closed-form*
 * function of `(lambda, k, D_l, W0, tau0)` -- the capillary length, the
 * kinetic coefficient, the steady planar velocity, the solute boundary-layer
 * width. Putting them next to the struct that holds the inputs is what makes
 * it possible to state the acceptance criterion as "measured == predicted"
 * rather than "measured == a number somebody wrote down once". The stepper
 * never reads these predictions; only the diagnostics do. That separation is
 * deliberate: a model that consulted its own oracle would not be a test.
 *
 * ### Units
 *
 * Lengths are in `W0`, times in `tau0`, so `dx` and `dt` are already
 * dimensionless and `D_l` is the dimensionless diffusivity
 * `D * tau0 / W0^2`. `W0 = tau0 = 1` is the normal choice; both are kept as
 * explicit members so a caller can check that a result is invariant under
 * rescaling them (it must be).
 *
 * ### Thin-interface asymptotics (Karma & Rappel 1998; Karma 2001;
 *     Echebarria, Folch, Karma & Plapp 2004)
 *
 * For the coupling `- lambda (1-phi^2)^2 U` used in equation (2) together
 * with the one-sided mobility `q(phi) = (1-phi)/2` and the anti-trapping
 * current of equation (3), the interface obeys
 *
 *     U_interface = - d0 * kappa - beta * V
 *
 * with
 *
 *     d0   = a1 * W0 / lambda
 *     beta = a1 * (tau0 / (lambda * W0)) * (1 - a2 * lambda * W0^2 / (tau0 * D_l))
 *     a1   = 5 sqrt(2) / 8 = 0.883883...,   a2 = 0.6267
 *
 * `beta = 0` is the usual "vanishing kinetics" choice `lambda = D_l / a2`.
 * Stage 1 deliberately does **not** use it: with `beta = 0` a planar front is
 * velocity-degenerate (see @ref planar_steady_velocity) and there is nothing
 * to measure. Stage 1 runs at finite `beta`, which turns the front velocity
 * into a direct, quantitative probe of the asymptotics.
 *
 * ### Where the elastic term attaches
 *
 * Equation (2) carries a term `- lambda_el (1-phi^2)^2 dF_el/dphi`.
 * @ref ModelParams::lambda_el and the field pointer on
 * `alloy_dendrite::Stepper` are the entire attachment surface; see
 * `step.hpp`, stage B, marked `ELASTIC HOOK`, and `elasticity.hpp` for
 * equations (5)-(7) and for what `lambda_el` means in physical units.
 *
 * @see Karma & Rappel, Phys. Rev. E 57, 4323 (1998)
 * @see Karma, Phys. Rev. Lett. 87, 115701 (2001)
 * @see Echebarria, Folch, Karma & Plapp, Phys. Rev. E 70, 061604 (2004)
 * @see Ramirez, Beckermann, Karma & Diepers, Phys. Rev. E 69, 051607 (2004)
 */

#include <cmath>

#include <openpfc/kernel/field/fd_stencils.hpp>
#include <cstdio>
#include <string>

namespace alloy_dendrite {

/// `a1 = 5 sqrt(2) / 8`, the capillary constant of the thin-interface limit.
inline constexpr double kA1 = 0.88388347648318440;
/// `a2 = 0.6267`, the kinetic constant of the thin-interface limit.
inline constexpr double kA2 = 0.6267;

/**
 * @brief Anti-trapping prefactor `a_t = +1 / (2 sqrt(2))` of equation (3).
 *
 * @details
 * The sign is positive, and the reason is worth writing down because the
 * intuitive argument gets it backwards. "The current pushes solute out of the
 * solid, `n` points into the solid, therefore the coefficient is negative" is
 * wrong: `a_t` is not fixed by a transport picture, it is fixed by the
 * requirement that the chemical potential stop varying inside the solid.
 *
 * Take the 1-D steady front (`xi = x - Vt`, solid at `xi < 0`, so
 * `phi' < 0` and `d_t phi = -V phi' > 0`), substitute the equilibrium
 * profile `phi' = -(1 - phi^2) / (sqrt(2) W0)`, and integrate the conserved
 * form of (3) once from the deep solid. With `A = 1 + (1-k) U`,
 *
 *     U'(xi) = -(A V / D_l) [ 1 - sqrt(2) a_t (1 + phi) ]
 *
 * At `phi = +1` the bracket is `1 - 2 sqrt(2) a_t`, which vanishes only for
 * `a_t = +1/(2 sqrt(2))`. Any other value leaves a solute gradient inside a
 * phase that has zero diffusivity -- the spurious trapping. So the
 * anti-trapping current is the term that makes the one-sided limit
 * self-consistent, and its sign follows from that cancellation, not from
 * which way the flux vector happens to point.
 *
 * `MODEL_SPEC.md` equation (3) has this right. @ref ModelParams::at_scale
 * still exposes it as a run-time multiplier, because "`k_eff` is flat in `V`
 * only for `at_scale = 1`" is a measurement worth making rather than a
 * property worth asserting.
 */
inline const double kAntiTrapCoeff = 1.0 / (2.0 * std::sqrt(2.0));

/**
 * @brief Squared-gradient floor below which `grad phi / |grad phi|` is taken
 *        to be zero.
 *
 * The anisotropy flux of equation (2) carries a spare factor of `|grad phi|`
 * and is therefore self-regularising; the anti-trapping current of equation
 * (3) is not -- it is `O(1)` in direction no matter how small the gradient
 * is. In the bulk the factor `d_t phi` that multiplies it is at round-off
 * level, so gating the *direction* at a gradient this small changes nothing
 * physical while removing the possibility of a `0/0`. Chosen `1e-24`, i.e.
 * `|grad phi| < 1e-12`, roughly the square root of double precision.
 */
inline constexpr double kGradNormFloor2 = 1.0e-24;

/**
 * @brief Physical parameters of equations (1)-(4).
 *
 * Defaults are the Stage-1 planar-verification point: isotropic, no thermal
 * feedback, `beta` comfortably nonzero so the planar velocity is a real
 * prediction. The 2-D dendrite driver overrides `eps4`, `M_c` and `D_th`.
 */
struct ModelParams {
  /// Interface width. The length unit; leave at 1 unless testing invariance.
  double W0 = 1.0;
  /// Relaxation time. The time unit; leave at 1 unless testing invariance.
  double tau0 = 1.0;
  /// Coupling constant of equation (2). Sets `d0 = a1 W0 / lambda`.
  double lambda = 1.0;
  /// Equilibrium partition coefficient `k = c_s / c_l`. Must be in `(0, 1)`.
  double k = 0.15;
  /// Liquid solute diffusivity in `W0^2 / tau0`.
  double D_l = 2.0;
  /// Thermal diffusivity in `W0^2 / tau0`. Typically `>> D_l`; 0 freezes (4).
  double D_th = 0.0;
  /// Thermal coupling `M_c` in equation (2). 0 makes the phase field
  /// isothermal even while equation (4) keeps releasing latent heat, which is
  /// what lets Stage 1 exercise the latent-heat balance without perturbing
  /// the solute problem it is measuring.
  double M_c = 0.0;
  /// Cubic anisotropy strength `eps4` of equation (1).
  double eps4 = 0.0;
  /**
   * @brief Multiplier on the anti-trapping current.
   *
   * `1` is the physical model. `0` switches the current off -- the classic
   * one-sided model, whose `k_eff` then drifts upward with velocity. `-1`
   * flips the sign, which doubles the spurious solid-side gradient instead
   * of cancelling it. Provided so that "the anti-trapping current is right"
   * is something the `k_eff` diagnostic *demonstrates* against two wrong
   * alternatives, rather than something this code asserts.
   */
  double at_scale = 1.0;
  /**
   * @brief Use the literal source term of `MODEL_SPEC.md` equation (3).
   *
   * The spec pairs the conservative left-hand side `d_t[P U]` with the
   * source `(1/2)(1 + (1-k) U) d_t phi`, which belongs to the
   * non-conservative form `P d_t U`. `false` (the default) uses the
   * consistent conservative source `(1/2) d_t phi`; `true` reproduces the
   * spec verbatim so the size of the resulting error can be measured. See
   * the file comment of `step.hpp` for the derivation and for what the
   * mismatch does to `k_eff`.
   */
  bool spec_source = false;
  /**
   * @brief Elastic coupling `lambda_el` of equation (2).
   *
   * Zero switches the elastic feedback off *bitwise*, which is what makes an
   * elastic-off/elastic-on comparison a controlled experiment rather than
   * two different programs. See `elasticity.hpp` for the derivation of
   * `lambda_el = lambda` when the stiffnesses are expressed in units of the
   * chemical free-energy scale `f_ref = L dT_0 / T_M`.
   */
  double lambda_el = 0.0;
  /// Advance equation (4). Off makes theta a frozen zero field and skips the
  /// thermal Laplacian entirely.
  bool evolve_theta = true;
};

/// Solute prefactor `P(phi) = ((1+k) - (1-k) phi) / 2` of equation (3).
/// `P = 1` in the liquid, `P = k` in the solid; strictly positive for
/// `k > 0`, so `U = psi / P` never divides by zero.
[[nodiscard]] inline double solute_prefactor(double k, double phi) noexcept {
  return 0.5 * ((1.0 + k) - (1.0 - k) * phi);
}

/// One-sided mobility `q(phi) = (1 - phi) / 2` of equation (3).
[[nodiscard]] inline double solute_mobility(double phi) noexcept {
  return 0.5 * (1.0 - phi);
}

/// Chemical capillary length `d0 = a1 W0 / lambda`.
[[nodiscard]] inline double capillary_length(const ModelParams &p) noexcept {
  return kA1 * p.W0 / p.lambda;
}

/**
 * @brief Thin-interface kinetic coefficient
 *        `beta = a1 (tau0 / (lambda W0)) (1 - a2 lambda W0^2 / (tau0 D_l))`.
 *
 * Zero at `lambda = D_l tau0 / (a2 W0^2)`; negative beyond that, which is
 * unphysical and where the model becomes ill-posed.
 */
[[nodiscard]] inline double kinetic_coefficient(const ModelParams &p) noexcept {
  return kA1 * (p.tau0 / (p.lambda * p.W0)) *
         (1.0 - kA2 * p.lambda * p.W0 * p.W0 / (p.tau0 * p.D_l));
}

/**
 * @brief Steady velocity of an isothermal planar front at supersaturation
 *        `Omega`: `V = (Omega - 1) / (k beta)`.
 *
 * @details
 * Sharp-interface derivation, one-sided model, front advancing into liquid of
 * far-field composition `c_inf`, reference `c_l^0` the equilibrium liquidus
 * composition at the (fixed) temperature, `Omega = (c_l^0 - c_inf) /
 * (c_l^0 (1-k)) = -U_inf`:
 *
 *  - a steady travelling wave forces the newly formed solid to carry exactly
 *    the far-field composition, `c_s = c_inf`;
 *  - local equilibrium at a planar interface gives `c_s = k c_l^i` and
 *    `c_l^i = c_l^0 (1 + (1-k) U_i)` with `U_i = -beta V`;
 *  - eliminating `c_s` and `c_l^i` gives `Omega = 1 + k beta V`.
 *
 * Two consequences worth stating out loud, because they decide the test
 * design:
 *
 *  1. At `beta = 0` the relation collapses to `Omega = 1` with `V`
 *     *undetermined* -- the classical degeneracy of planar one-sided growth.
 *     A velocity test at vanishing kinetics measures nothing.
 *  2. The prediction is a ratio of two small numbers: `Omega - 1` is
 *     `k beta V`, a fraction of a percent at the velocities where the
 *     thin-interface limit is valid. Getting `V` right to a few percent
 *     therefore requires the *whole* asymptotic relation to be right to a few
 *     percent, which is what makes it a strong test rather than a weak one.
 */
[[nodiscard]] inline double planar_steady_velocity(const ModelParams &p,
                                                   double omega) noexcept {
  return (omega - 1.0) / (p.k * kinetic_coefficient(p));
}

/// Inverse of @ref planar_steady_velocity: the `Omega` that drives a planar
/// front at `V`. Drivers take `V` as the input and derive `Omega`, because
/// `V` is the quantity whose magnitude has to stay inside the thin-interface
/// window (`W0 V / D_l << 1`).
[[nodiscard]] inline double planar_supersaturation(const ModelParams &p,
                                                   double velocity) noexcept {
  return 1.0 + p.k * kinetic_coefficient(p) * velocity;
}

/// Solute boundary-layer width `D_l / V` of the steady planar profile.
[[nodiscard]] inline double boundary_layer_width(const ModelParams &p,
                                                 double velocity) noexcept {
  return p.D_l / velocity;
}

/// Interface Peclet number `W0 V / D_l`. The thin-interface expansion is an
/// expansion in this; the drivers warn above ~0.1.
[[nodiscard]] inline double interface_peclet(const ModelParams &p,
                                             double velocity) noexcept {
  return p.W0 * velocity / p.D_l;
}

/**
 * @brief Explicit-Euler stability ceiling used by the drivers as a guard.
 *
 * Three competing limits on a `d`-dimensional grid:
 *  - phase-field diffusion `dx^2 / (2 d W0^2 / tau0)`,
 *  - solute diffusion `dx^2 / (2 d D_l)`,
 *  - thermal diffusion `dx^2 / (2 d D_th)`.
 *
 * plus a reaction guard `tau0 / 4`. The linearised rate of the double well
 * at `phi = +/-1` is `2 / tau0`, so `dt < tau0 / 2` is the bare limit and a
 * factor two of margin covers the `lambda (1-phi^2)^2 U` term at the
 * couplings used here. PR #103 arrives at the same guard empirically
 * (`kDtOverTau = 0.05`, i.e. twenty times more conservative than this) and
 * adds a third one this function cannot know about: an *interface* CFL
 * `dx / V`, so that the front does not cross a cell in one step and leave
 * the discrete `d_t phi` that feeds the anti-trapping current
 * under-resolved. The drivers check that one separately, where `V` is known.
 *
 * @note The anisotropy of equation (1) does not change the leading term:
 *       `W(n)^2 / tau(n) = W0^2 a_s^2 / (tau0 a_s^2) = W0^2 / tau0` exactly,
 *       which is one of the reasons `tau = tau0 a_s^2` is the right pairing
 *       for `W = W0 a_s`. The *anisotropy flux* `A_i` does add an effective
 *       diffusivity of order `4 eps4 W0^2 / tau0`, so at large `eps4` and
 *       small `D_l` this estimate is optimistic; at the shipped parameters
 *       (`eps4 = 0.2`, `D_l = 2`) the solute limit is the binding one by a
 *       factor of ten and the point does not arise.
 */
/**
 * @brief Nyquist eigenvalue of the central second-derivative stencil of
 *        order @p order, in units of `1/dx^2`.
 *
 * The symbol of the stencil is
 * `Lhat(k) = (1/dx^2) [c_0 + 2 sum_m c_m cos(m k dx)] / denom`, and the
 * explicit Euler stability bound is set by its largest magnitude, which for
 * a central second derivative is at the Nyquist mode `k dx = pi` where
 * `cos(m pi) = (-1)^m`. Returns `|Lhat| dx^2`, i.e. 4 at second order,
 * rising monotonically to `pi^2 = 9.87` as the order goes to infinity.
 *
 * Reading it out of the same table the stepper differentiates with is
 * deliberate: a hand-tabulated copy is a table that can drift out of sync
 * with the stencil it is supposed to describe, and the whole point of this
 * function is to be right about the stencil actually in use.
 */
[[nodiscard]] inline double d2_nyquist_eigenvalue(int order) noexcept {
  pfc::field::fd::EvenCentralD2View v{};
  if (!pfc::field::fd::lookup_even_central_d2(order, &v)) {
    return 4.0; // unsupported order; the stepper rejects it separately
  }
  double sum = static_cast<double>(v.coeffs[0]);
  for (int m = 1; m <= v.half_width; ++m) {
    sum += 2.0 * static_cast<double>(v.coeffs[m]) * ((m % 2 == 0) ? 1.0 : -1.0);
  }
  return std::fabs(sum / static_cast<double>(v.denom));
}

/**
 * @brief Largest explicit-Euler step that is stable for @p p on a grid of
 *        spacing @p dx in @p dim dimensions with a stencil of order @p order.
 *
 * @note **The order matters and it used to be ignored.** This function
 *       returned the second-order von Neumann bound `dx^2/(2 d D)` for every
 *       stencil, which corresponds to a Nyquist eigenvalue of 4. The real
 *       one grows with the order -- 4, 5.33, 6.04, 6.42, 6.68, 6.87 for
 *       orders 2 to 12, tending to `pi^2` -- so the bound was too generous
 *       by a factor of 1.7 at order 12. Every run in the campaign used
 *       `dt_safety = 0.2`, a fivefold margin, and none of them was affected;
 *       a run at `dt_safety = 0.8` and order 12 would have been unstable
 *       while this function asserted it was inside the limit, which is
 *       exactly the failure this application refuses to have elsewhere.
 */
[[nodiscard]] inline double explicit_dt_limit(const ModelParams &p, double dx,
                                              int dim, int order = 2) noexcept {
  const double lam = d2_nyquist_eigenvalue(order);
  // |1 - dt D lambda| <= 1  =>  dt <= 2 / (D lambda), with lambda summed
  // over `dim` axes: lambda_total = dim * lam / dx^2.
  const double denom = 0.5 * lam * static_cast<double>(dim);
  double lim = dx * dx / (denom * p.W0 * p.W0 / p.tau0);
  lim = std::fmin(lim, dx * dx / (denom * p.D_l));
  if (p.evolve_theta && p.D_th > 0.0) {
    lim = std::fmin(lim, dx * dx / (denom * p.D_th));
  }
  return std::fmin(lim, 0.25 * p.tau0);
}

/**
 * @brief Interfacial-stiffness ceiling on `eps4` in 2-D.
 *
 * With `a_s = 1 + eps4 cos 4 theta` the 2-D stiffness is
 * `a_s + a_s'' = 1 - 15 eps4 cos 4 theta`, which first vanishes at
 * `eps4 = 1/15`. Beyond it the equilibrium shape has missing orientations and
 * the smooth-tip selection theory the Stage-2 measurement is compared against
 * does not apply -- the run will still produce a number, which is exactly why
 * the ceiling is worth naming.
 */
inline constexpr double kEps4StiffnessLimit = 1.0 / 15.0;

/// Minimum of the normalised `a_s` over orientation, `1 - eps4` in 2-D and
/// `1 - 5 eps4 / 3` in 3-D. Must stay positive; see `step.hpp`.
[[nodiscard]] inline double anisotropy_min(double eps4, int dim) noexcept {
  return (dim == 3) ? (1.0 - 5.0 * eps4 / 3.0) : (1.0 - eps4);
}

/// Human-readable one-line dump of the derived quantities, for run headers.
[[nodiscard]] inline std::string derived_summary(const ModelParams &p) {
  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "d0=%.6g beta=%.6g lambda=%.6g k=%.6g D_l=%.6g "
                "lambda_beta0=%.6g eps4=%.6g at_scale=%.6g spec_source=%d",
                capillary_length(p), kinetic_coefficient(p), p.lambda, p.k, p.D_l,
                p.D_l * p.tau0 / (kA2 * p.W0 * p.W0), p.eps4, p.at_scale,
                p.spec_source ? 1 : 0);
  return std::string(buf);
}

} // namespace alloy_dendrite
