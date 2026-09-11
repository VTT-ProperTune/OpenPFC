// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file ics.hpp
 * @brief Analytic initial distributions for the validation ladder of #84.
 *
 * @details
 * Every function here is a pure `double(x, vx, vy)` with no dependence on
 * the field layout, the decomposition or the stepper. That is deliberate:
 * the initial condition is the thing a benchmark is *defined* by, so it has
 * to be readable, testable and quotable on its own, and it has to be
 * possible to integrate it analytically when checking the moment deposition.
 *
 * All distributions are normalised so that `int f dvx dvy = n(x)/n_0`, in
 * the units of `parameters.hpp` (velocities in `c`, so a thermal velocity is
 * a number like 0.05 and a drift is a number like 0.2).
 *
 * ## The one thing to be careful about
 *
 * These are *continuum* initial conditions sampled at cell centres, not
 * particle loads. A Maxwellian sampled on a finite velocity grid has a
 * discrete zeroth moment that differs from 1 by the truncation error of the
 * quadrature plus the tail outside `[-v_max, v_max]`. For `v_max / v_th >= 6`
 * that difference is below `1e-16` and the distinction is invisible; below
 * about 4 it is not, and it shows up as a spurious net charge. @ref
 * maxwellian_truncation_error gives the analytic size of the tail so that a
 * run can report it rather than absorb it, and the drivers refuse a
 * configuration where it exceeds a threshold.
 */

#include <cmath>
#include <stdexcept>

namespace vlasov::ics {

/// `1/sqrt(2 pi)`, the 1-D Maxwellian prefactor before the `1/v_th`.
inline const double kInvSqrt2Pi = 1.0 / std::sqrt(2.0 * std::acos(-1.0));

/**
 * @brief Isotropic 2-D Maxwellian of thermal velocity @p vth, drifting at
 *        `(ux, uy)`, at density @p density.
 *
 * `f = n/(2 pi vth^2) exp(-[(vx-ux)^2 + (vy-uy)^2]/(2 vth^2))`, which
 * integrates to `n` over the whole velocity plane.
 */
[[nodiscard]] inline double maxwellian(double vx, double vy, double vth,
                                       double density = 1.0, double ux = 0.0,
                                       double uy = 0.0) noexcept {
  const double a = (vx - ux) / vth;
  const double b = (vy - uy) / vth;
  const double pref = density / (2.0 * std::acos(-1.0) * vth * vth);
  return pref * std::exp(-0.5 * (a * a + b * b));
}

/**
 * @brief Bi-Maxwellian with different temperatures along `x` and `y`.
 *
 * The Weibel driver. The anisotropy `A = (vthy/vthx)^2 - 1` is the parameter
 * the transverse dispersion relation is written in, and it is positive when
 * the distribution is hotter across the wave vector than along it, which is
 * the unstable case.
 */
[[nodiscard]] inline double bi_maxwellian(double vx, double vy, double vthx,
                                          double vthy,
                                          double density = 1.0) noexcept {
  const double a = vx / vthx;
  const double b = vy / vthy;
  const double pref = density / (2.0 * std::acos(-1.0) * vthx * vthy);
  return pref * std::exp(-0.5 * (a * a + b * b));
}

/**
 * @brief Temperature ratio `A = T_y/T_x = (vthy/vthx)^2` of @ref bi_maxwellian.
 *
 * **This is the `A` the transverse dispersion relation is written in**
 * (`openpfc_apps/plasma_dispersion.hpp`, `BiMaxwellian::anisotropy`), and
 * the two must agree or the Weibel growth rate is compared against the
 * wrong oracle. An earlier revision of this header defined `anisotropy` as
 * `A - 1` -- the *excess* -- which is the form the marginal-stability
 * condition `k^2 c^2 = omega_pe^2 (A - 1)` is usually quoted in. Two
 * functions of the same name differing by one is precisely the silent
 * mismatch that produces a confident wrong number, so the ratio is now the
 * only thing called `A` and the excess has its own name.
 */
[[nodiscard]] inline double temperature_ratio(double vthx,
                                              double vthy) noexcept {
  const double r = vthy / vthx;
  return r * r;
}

/// `A - 1`, the quantity the Weibel cutoff `k^2 c^2 = omega_pe^2 (A - 1)` is
/// linear in. Positive means unstable: hotter across the wave vector than
/// along it.
[[nodiscard]] inline double anisotropy_excess(double vthx,
                                              double vthy) noexcept {
  return temperature_ratio(vthx, vthy) - 1.0;
}

/**
 * @brief Two counter-streaming warm beams along `v_x`, the electrostatic
 *        two-stream configuration.
 *
 * Equal halves at `+u` and `-u`. The instability is electrostatic and lives
 * in `E_x`, so this is the benchmark for the Vlasov-Poisson reduction.
 */
[[nodiscard]] inline double two_stream_x(double vx, double vy, double vth,
                                         double u,
                                         double density = 1.0) noexcept {
  return 0.5 * maxwellian(vx, vy, vth, density, +u, 0.0) +
         0.5 * maxwellian(vx, vy, vth, density, -u, 0.0);
}

/**
 * @brief Two counter-streaming beams along `v_y`, the electromagnetic
 *        filamentation (cold Weibel) configuration.
 *
 * The drift is *across* the wave vector `k x-hat`, so the free energy is in
 * the transverse current and the unstable mode is `B_z` -- electromagnetic,
 * not electrostatic. That difference from @ref two_stream_x is the whole
 * point of stage 4, and it is why the two look almost identical here and
 * behave completely differently.
 */
[[nodiscard]] inline double two_stream_y(double vx, double vy, double vth,
                                         double u,
                                         double density = 1.0) noexcept {
  return 0.5 * maxwellian(vx, vy, vth, density, 0.0, +u) +
         0.5 * maxwellian(vx, vy, vth, density, 0.0, -u);
}

/// Density modulation `1 + amplitude cos(k x)`, the standard perturbation.
[[nodiscard]] inline double density_perturbation(double x, double k,
                                                 double amplitude) noexcept {
  return 1.0 + amplitude * std::cos(k * x);
}

/**
 * @brief Fraction of a Maxwellian lying outside `|v| > v_max` in one
 *        dimension: `erfc(v_max/(vth sqrt 2))`.
 *
 * The analytic size of the truncation the finite velocity box imposes.
 * Reported by every run so that the velocity boundary is a measured quantity
 * rather than an assumption; see the boundary discussion in #84.
 */
[[nodiscard]] inline double maxwellian_truncation_error(double v_max,
                                                        double vth) noexcept {
  return std::erfc(v_max / (vth * std::sqrt(2.0)));
}

/**
 * @brief Refuse a velocity box that truncates more than @p tol of the
 *        distribution.
 *
 * A truncated Maxwellian is not a Maxwellian: its moments are wrong, the
 * deposited charge is wrong, and because the error is smooth and small it
 * produces a run that looks healthy and damps at the wrong rate. Ten
 * thermal widths costs nothing and removes the failure mode entirely, so
 * the default threshold is strict.
 */
inline void require_resolved_tail(double v_max, double vth,
                                  double tol = 1.0e-12) {
  const double e = maxwellian_truncation_error(v_max, vth);
  if (e > tol) {
    throw std::invalid_argument(
        "velocity box truncates " + std::to_string(e) +
        " of the distribution (tolerance " + std::to_string(tol) +
        "): increase v_max or reduce v_th. v_max/v_th = " +
        std::to_string(v_max / vth));
  }
}

} // namespace vlasov::ics
