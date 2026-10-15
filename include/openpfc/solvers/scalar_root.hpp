// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file scalar_root.hpp
 * @brief Pointwise safeguarded Newton for one monotone scalar equation.
 *
 * @details
 * Solves `f(x) = 0` on a caller bracket `[lo, hi]` where `f(lo)` and `f(hi)`
 * have opposite signs (or one is zero). Each iteration takes the Newton step
 * from the current point; when that step leaves the bracket, or `f'` is zero
 * or not finite, it bisects instead. The bracket
 * shrinks on every evaluation, so the method cannot diverge.
 *
 * Intended for the per-cell inversions of phase-field and constitutive laws,
 * such as finding the diffusion potential `mu` from a concentration. It does
 * not allocate or throw and is callable from device code.
 *
 * Stops when `|f| <= f_tol`, when the next step is at most
 * `x_tol * (1 + |x|)`, or after `max_iterations` evaluations of `f`
 * (the two bracket ends included). The result is the last evaluated point
 * and its residual. Systems of equations belong to `newton.hpp`.
 */

#include <cmath>

#include <openpfc/kernel/data/host_device.hpp>

namespace pfc::solvers {

enum class ScalarRootStatus { Converged, IterationLimit, NoBracket, NonFinite };

struct ScalarRootControls {
  double f_tol = 0.0;       ///< absolute residual tolerance
  double x_tol = 1e-14;     ///< relative step tolerance
  int max_iterations = 200; ///< evaluations of f
};

struct ScalarRootResult {
  double x = 0.0;
  double f = 0.0;
  double lo = 0.0, hi = 0.0; ///< final bracket
  int iterations = 0;
  ScalarRootStatus status = ScalarRootStatus::IterationLimit;
  OPENPFC_HD bool converged() const noexcept {
    return status == ScalarRootStatus::Converged;
  }
};

/// Value and derivative returned by the caller's function.
struct ScalarValue {
  double f = 0.0;
  double df = 0.0;
};

/**
 * @brief Root of `fun` in `[lo, hi]` starting from `x0` (clamped).
 *
 * `fun(x)` returns `ScalarValue{f(x), f'(x)}`. The function need not be
 * increasing; only the sign change on the bracket is required.
 */
template <class Fun>
OPENPFC_HD ScalarRootResult scalar_root(Fun &&fun, double lo, double hi, double x0,
                                        ScalarRootControls c = {}) noexcept {
  ScalarRootResult r;
  if (!(lo <= hi) || !std::isfinite(lo) || !std::isfinite(hi)) {
    r.status = ScalarRootStatus::NoBracket;
    return r;
  }
  ScalarValue vlo = fun(lo), vhi = fun(hi);
  r.iterations = 2;
  if (!std::isfinite(vlo.f) || !std::isfinite(vhi.f)) {
    r.status = ScalarRootStatus::NonFinite;
    return r;
  }
  // A root on the bracket end, to within f_tol, is accepted there: an
  // analytic bracket often has the root exactly on one end.
  const bool at_lo = std::abs(vlo.f) <= c.f_tol || vlo.f == 0.0;
  if (at_lo || std::abs(vhi.f) <= c.f_tol || vhi.f == 0.0) {
    r.x = at_lo ? lo : hi;
    r.f = at_lo ? vlo.f : vhi.f;
    r.lo = r.hi = r.x;
    r.status = ScalarRootStatus::Converged;
    return r;
  }
  if ((vlo.f > 0.0) == (vhi.f > 0.0)) {
    r.status = ScalarRootStatus::NoBracket;
    return r;
  }
  const bool rising = vhi.f > 0.0; // f(lo) < 0 < f(hi)
  double x = x0 < lo ? lo : (x0 > hi ? hi : x0);
  while (r.iterations < c.max_iterations) {
    const ScalarValue v = fun(x);
    ++r.iterations;
    r.x = x;
    r.f = v.f;
    if (!std::isfinite(v.f)) {
      r.status = ScalarRootStatus::NonFinite;
      break;
    }
    if (std::abs(v.f) <= c.f_tol || v.f == 0.0) {
      r.status = ScalarRootStatus::Converged;
      break;
    }
    if ((v.f > 0.0) == rising)
      hi = x;
    else
      lo = x;
    double xn =
        (std::isfinite(v.df) && v.df != 0.0) ? x - v.f / v.df : 0.5 * (lo + hi);
    if (!(xn > lo && xn < hi)) xn = 0.5 * (lo + hi);
    // A negligible step, or a bracket collapsed to adjacent doubles, ends
    // the solve at the last evaluated point, so r.f belongs to r.x.
    if (std::abs(xn - x) <= c.x_tol * (1.0 + std::abs(x)) || !(lo < xn && xn < hi)) {
      r.status = ScalarRootStatus::Converged;
      break;
    }
    x = xn;
  }
  r.lo = lo;
  r.hi = hi;
  return r;
}

} // namespace pfc::solvers
