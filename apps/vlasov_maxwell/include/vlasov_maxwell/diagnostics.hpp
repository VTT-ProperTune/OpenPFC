// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file diagnostics.hpp
 * @brief The conservation ledger, and the append-only CSV it is written to.
 *
 * @details
 * ## Why a ledger and not a list of numbers
 *
 * The Vlasov-Maxwell system conserves particle number, momentum and energy,
 * and it conserves every Casimir `int G(f)` for any `G` -- in particular
 * `int |f|`, `int f^2` and the entropy `-int f ln f`. None of those are
 * approximately conserved. They are exact invariants of the continuous flow,
 * and the *only* reason a computed one moves is numerical error.
 *
 * That makes them a measurement of the scheme rather than a property of the
 * run, and it is why they are collected in one place with their expected
 * behaviour written next to them:
 *
 *  - **Particle number** is conserved to interpolation error by the
 *    semi-Lagrangian steps and exactly by the spectral shift.
 *  - **Energy** exchanges between the kinetic and field parts, so neither
 *    half is conserved and the *total* is. That exchange is the physics --
 *    magnetic energy growing while velocity anisotropy decays is the Weibel
 *    instability -- so the ledger keeps the halves as well as the sum.
 *  - **Entropy** is conserved analytically. Its drift is therefore a direct
 *    measurement of numerical diffusion, which is the error that filamenting
 *    velocity structure produces and the one no other diagnostic sees.
 *  - **`min f`** should be zero and will not be: Lagrange interpolation of
 *    order above one is not positivity-preserving. The size of the
 *    negativity is reported rather than clipped, because clipping destroys
 *    the conservation the rest of this table is measuring.
 *  - **The Gauss residual** is the sharpest of all; see `parameters.hpp`.
 *
 * ## The drift convention
 *
 * Every conserved quantity is reported both as its value and as a relative
 * drift from `t = 0`. The drift is normalised by the *initial* value where
 * that is nonzero and by a stated scale where it is not -- total momentum
 * starts at zero in every benchmark here, so a relative drift would be
 * meaningless and the absolute value is reported against the scale of the
 * individual kinetic and field contributions instead. Reporting a relative
 * drift against a vanishing baseline is a good way to publish a number that
 * looks catastrophic or perfect at random.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <vlasov_maxwell/maxwell.hpp>
#include <vlasov_maxwell/moments.hpp>
#include <vlasov_maxwell/parameters.hpp>

namespace vlasov {

/// One row of the conservation ledger.
struct Ledger {
  double t{0.0};
  int step{0};

  // ---- kinetic, summed over species -------------------------------------
  double number{0.0};
  double kinetic_energy{0.0};
  double momentum_x{0.0};
  double momentum_y{0.0};
  double l1{0.0};
  double l2{0.0};
  double entropy{0.0};
  double f_min{0.0};
  double f_max{0.0};

  // ---- fields -----------------------------------------------------------
  /// `0.5 int E_x^2 dx`: the electrostatic energy, the Landau observable.
  double energy_ex{0.0};
  /// `0.5 int (E_y^2 + B_z^2) dx`: the electromagnetic energy, the Weibel
  /// observable. Kept apart from `energy_ex` because the whole point of
  /// stage 4 is that one of them grows when the other does not.
  double energy_em{0.0};
  /// `0.5 int B_z^2 dx` alone. The Weibel growth rate is fitted to this.
  double energy_bz{0.0};
  double field_momentum_x{0.0};
  /**
   * @brief `|hat E_x(m)|` and `|hat B_z(m)|` at the seeded mode.
   *
   * The observable every linear benchmark should actually be fitted to.
   * Total field energy is the sum over all modes, and in an instability run
   * the mode that eventually dominates is the fastest-growing one, which is
   * not necessarily the one that was seeded. Fitting the total therefore
   * measures whichever mode won, at whatever amplitude it started from --
   * which is how a growth rate comes out 79% high while every conservation
   * diagnostic reads 1e-14. Fitting the seeded mode's own amplitude
   * measures the thing linear theory makes a prediction about.
   */
  double mode_ex{0.0};
  double mode_bz{0.0};
  /**
   * @brief `max(|E_x|, |E_y|, |B_z|)`: the scale the Gauss residual means
   *        something against.
   *
   * `gauss_residual` divides by `max|rho|`, which is the right
   * normalisation for an electrostatic run and a 0/0 for an
   * electromagnetic one -- a Weibel plasma stays neutral to `1e-5`, so the
   * relative form reads 0.27 while the absolute residual is `4e-6` against
   * a field of `8e-2`. Reporting the scale as its own column lets a reader
   * form the ratio that is meaningful for the run in front of them instead
   * of trusting one that was chosen in advance.
   */
  double field_scale{0.0};

  // ---- totals -----------------------------------------------------------
  double total_energy{0.0};
  double total_momentum_x{0.0};

  // ---- constraint and boundary -------------------------------------------
  double gauss_residual{0.0};
  double gauss_abs_residual{0.0};
  double net_charge{0.0};
  /// `max|f|` on the velocity-boundary faces, relative to `max|f|`.
  double boundary_ratio{0.0};
  /// Fraction of particle number within one thermal width of the boundary.
  double boundary_fraction{0.0};

  // ---- drifts from t = 0 --------------------------------------------------
  double d_number{0.0};
  double d_energy{0.0};
  double d_l1{0.0};
  double d_l2{0.0};
  double d_entropy{0.0};
  /// Absolute, not relative: total momentum starts at zero in every
  /// benchmark here, so a relative drift would divide by nothing.
  double d_momentum_x{0.0};
};

/**
 * @brief `(1/2) int g^2 dx` for one field component.
 *
 * `maxwell.hpp` offers the *total* field energy and the transverse pair;
 * the ledger needs the electrostatic and magnetic parts on their own,
 * because the whole content of stage 4 is that one of them grows while the
 * other does not. Same quadrature, one component.
 */
[[nodiscard]] inline double component_energy(const SpectralLine1D &line,
                                             const std::vector<double> &g) {
  double s = 0.0;
  for (const double v : g) s += v * v;
  return 0.5 * s * line.dx();
}

/// Relative change of @p now from @p ref, or the absolute change when @p ref
/// is too small for a ratio to mean anything.
[[nodiscard]] inline double drift(double now, double ref,
                                  double floor = 1.0e-300) noexcept {
  const double a = std::fabs(ref);
  return (a > floor) ? (now - ref) / a : (now - ref);
}

/**
 * @brief `|hat g(m)|`, the amplitude of a single Fourier mode.
 *
 * Normalised so that a field `A cos(k_m x)` gives exactly `A`. Two factors
 * are needed and an earlier revision had only one of them:
 *
 *  - `SpectralLine1D::forward` is the **unnormalised** DFT, so a bin
 *    carries `N` times the Fourier coefficient. Missing this made every
 *    reported amplitude `N` times too large -- harmlessly for the growth
 *    and damping *rates*, since a constant factor cancels in
 *    `d ln|E| / dt`, and not at all harmlessly for any figure or table
 *    that quotes a field strength.
 *  - a real signal splits its amplitude between the `+m` and `-m` bins,
 *    so the one-sided value is half.
 */
[[nodiscard]] inline double mode_amplitude(const SpectralLine1D &line,
                                           const std::vector<double> &g,
                                           int m) {
  const auto h = line.forward(g);
  if (m < 0 || static_cast<std::size_t>(m) >= h.size()) return 0.0;
  const double n = static_cast<double>(h.size());
  return 2.0 * std::abs(h[static_cast<std::size_t>(m)]) / n;
}

/**
 * @brief Assemble a ledger row from the per-species moments and the fields.
 *
 * @param p        grid and species
 * @param line     the spectral line used for the field integrals
 * @param mom      one @ref VelocityMoments per species, in `p.species` order
 * @param src      the deposited sources, for the net charge
 * @param fields   the current field state
 * @param gauss    the measured Gauss residual
 */
[[nodiscard]] inline Ledger
make_ledger(const SimParams &p, const SpectralLine1D &line,
            const std::vector<VelocityMoments> &mom, const Sources &src,
            const FieldState &fields, const GaussDiagnostic &gauss, double t,
            int step, int mode = 1) {
  Ledger L;
  L.t = t;
  L.step = step;
  for (std::size_t i = 0; i < mom.size() && i < p.species.size(); ++i) {
    const Species &s = p.species[i];
    L.number += mom[i].number;
    L.kinetic_energy += kinetic_energy(p, s, mom[i]);
    const auto pm = momentum(p, s, mom[i]);
    L.momentum_x += pm[0];
    L.momentum_y += pm[1];
    L.l1 += mom[i].l1;
    L.l2 += mom[i].l2;
    L.entropy += mom[i].entropy;
    L.f_min = std::fmin(L.f_min, mom[i].f_min);
    L.f_max = std::fmax(L.f_max, mom[i].f_max);
    L.boundary_ratio = std::fmax(L.boundary_ratio, mom[i].face_ratio());
    L.boundary_fraction = std::fmax(L.boundary_fraction, mom[i].boundary_fraction);
  }
  L.energy_ex = component_energy(line, fields.Ex);
  L.energy_bz = component_energy(line, fields.Bz);
  L.energy_em = transverse_energy(line, fields.Ey, fields.Bz);
  L.field_momentum_x = field_momentum_x(line, fields);
  L.mode_ex = mode_amplitude(line, fields.Ex, mode);
  L.mode_bz = mode_amplitude(line, fields.Bz, mode);
  for (std::size_t i = 0; i < fields.Ex.size(); ++i) {
    L.field_scale = std::fmax(
        L.field_scale,
        std::fmax(std::fabs(fields.Ex[i]),
                  std::fmax(std::fabs(fields.Ey[i]), std::fabs(fields.Bz[i]))));
  }
  L.total_energy = L.kinetic_energy + L.energy_ex + L.energy_em;
  L.total_momentum_x = L.momentum_x + L.field_momentum_x;
  L.gauss_residual = gauss.residual;
  L.gauss_abs_residual = gauss.abs_residual;
  L.net_charge = src.net_charge;
  return L;
}

/// Fill the drift columns of @p now against the `t = 0` row @p ref.
inline void set_drifts(Ledger &now, const Ledger &ref) noexcept {
  now.d_number = drift(now.number, ref.number);
  now.d_energy = drift(now.total_energy, ref.total_energy);
  now.d_l1 = drift(now.l1, ref.l1);
  now.d_l2 = drift(now.l2, ref.l2);
  now.d_entropy = drift(now.entropy, ref.entropy);
  now.d_momentum_x = now.total_momentum_x - ref.total_momentum_x;
}

/// Header of the time-series CSV. One place, so a column can never be added
/// to the row without being added here.
[[nodiscard]] inline const char *ledger_header() {
  return "run_id,step,t,"
         "number,kinetic_energy,momentum_x,momentum_y,l1,l2,entropy,"
         "f_min,f_max,"
         "energy_ex,energy_em,energy_bz,field_momentum_x,mode_ex,mode_bz,"
         "field_scale,"
         "total_energy,total_momentum_x,"
         "gauss_residual,gauss_abs_residual,net_charge,"
         "boundary_ratio,boundary_fraction,"
         "d_number,d_energy,d_l1,d_l2,d_entropy,d_momentum_x";
}

/// Format one row. `%.17g` on the conserved quantities: a drift of `1e-15`
/// is the result, so printing six digits would throw it away.
[[nodiscard]] inline std::string ledger_row(const std::string &run_id,
                                            const Ledger &L) {
  char buf[2048];
  std::snprintf(
      buf, sizeof(buf),
      "%s,%d,%.10g,"
      "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
      "%.10g,%.10g,"
      "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.10g,"
      "%.17g,%.17g,"
      "%.6e,%.6e,%.6e,"
      "%.6e,%.6e,"
      "%.6e,%.6e,%.6e,%.6e,%.6e,%.6e",
      run_id.c_str(), L.step, L.t, L.number, L.kinetic_energy, L.momentum_x,
      L.momentum_y, L.l1, L.l2, L.entropy, L.f_min, L.f_max, L.energy_ex,
      L.energy_em, L.energy_bz, L.field_momentum_x, L.mode_ex, L.mode_bz,
      L.field_scale, L.total_energy,
      L.total_momentum_x, L.gauss_residual, L.gauss_abs_residual, L.net_charge,
      L.boundary_ratio, L.boundary_fraction, L.d_number, L.d_energy, L.d_l1,
      L.d_l2, L.d_entropy, L.d_momentum_x);
  return std::string(buf);
}

/**
 * @brief Least-squares fit of `ln y = a + g t` over a window, returning `g`.
 *
 * The growth- or damping-rate estimator for every linear benchmark. A fit
 * rather than a two-point ratio because the field energy of a damped
 * Landau wave *oscillates* underneath its exponential envelope, so two
 * points can land anywhere between the envelope and zero. Fitting `ln`
 * of the energy and halving gives the amplitude rate.
 *
 * @param t,y   the series; entries with `y <= 0` are skipped
 * @param t0,t1 the fit window in time
 * @return the slope, or NaN if fewer than three points fall in the window
 */
[[nodiscard]] inline double fit_exponential_rate(const std::vector<double> &t,
                                                 const std::vector<double> &y,
                                                 double t0, double t1) {
  double n = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
  for (std::size_t i = 0; i < t.size() && i < y.size(); ++i) {
    if (t[i] < t0 || t[i] > t1 || !(y[i] > 0.0) || !std::isfinite(y[i])) {
      continue;
    }
    const double x = t[i];
    const double l = std::log(y[i]);
    n += 1.0;
    sx += x;
    sy += l;
    sxx += x * x;
    sxy += x * l;
  }
  if (n < 3.0) {
    return std::nan("");
  }
  const double den = n * sxx - sx * sx;
  if (std::fabs(den) < 1.0e-300) {
    return std::nan("");
  }
  return (n * sxy - sx * sy) / den;
}

/**
 * @brief Exponential rate from the **envelope** of a damped or growing
 *        oscillation.
 *
 * `fit_exponential_rate` regresses `ln y` over every sample, which is
 * unbiased only if the window happens to span a whole number of
 * half-periods. A Landau-damped mode is `|E| ~ e^{gamma t}|cos(omega t +
 * phi)|`, so `ln|E| = gamma t + ln|cos|`, and a window covering 7.2
 * half-periods carries the leftover 0.2 of a period as a slope error.
 * Measured, that bias was **-9.9% on the Landau damping rate, and it did
 * not move at all** between `n_vx = 64` and 512 or between interpolation
 * orders 3 and 9 -- which is exactly how one can tell it is not a
 * discretisation error: a numerical error that is indifferent to the
 * discretisation is not numerical.
 *
 * Fitting the local maxima removes it: the peaks of `|cos|` all sit at 1,
 * so the envelope is a clean exponential. The cost is a factor of `pi`
 * fewer points, which is irrelevant next to a systematic bias.
 *
 * @return the fitted rate, or NaN if fewer than three peaks lie in the
 *         window -- which is itself the useful signal that the run is too
 *         short to measure a rate at this frequency.
 */
[[nodiscard]] inline double fit_envelope_rate(const std::vector<double> &t,
                                              const std::vector<double> &y,
                                              double t0, double t1) {
  std::vector<double> pt, py;
  for (std::size_t i = 1; i + 1 < t.size() && i + 1 < y.size(); ++i) {
    if (t[i] < t0 || t[i] > t1) continue;
    if (!(y[i] > 0.0) || !std::isfinite(y[i])) continue;
    if (y[i] >= y[i - 1] && y[i] >= y[i + 1]) {
      pt.push_back(t[i]);
      py.push_back(y[i]);
    }
  }
  if (pt.size() < 3) return std::nan("");
  return fit_exponential_rate(pt, py, pt.front(), pt.back());
}

/**
 * @brief A fit window that contains the exponential phase and nothing else.
 *
 * An instability seeded at `1e-5` and growing at `0.3` reaches order one
 * by `t ~ 40` and then saturates. A window fixed in advance as "the first
 * half of the run" therefore measures the growth *and* the saturation, and
 * reports a rate that is low by whatever fraction of the window is flat --
 * measured, **-64% on the two-stream case**.
 *
 * The window is instead chosen from the data: it starts where the mode has
 * grown by @p rise above its seed, so the initial transient in which the
 * seeded mode redistributes over the eigenmodes is excluded, and ends
 * where the mode reaches @p ceiling of its own maximum, which is below any
 * plausible saturation.
 *
 * @return `{t0, t1}`, or `{NaN, NaN}` if no such window exists -- which
 *         means the run never grew, and a rate should not be quoted.
 */
[[nodiscard]] inline std::array<double, 2>
auto_growth_window(const std::vector<double> &t, const std::vector<double> &y,
                   double rise = 5.0, double ceiling = 0.2) {
  const std::array<double, 2> none{std::nan(""), std::nan("")};
  if (t.size() < 4 || y.size() < 4) return none;
  double y0 = 0.0;
  for (const double v : y) {
    if (v > 0.0) {
      y0 = v;
      break;
    }
  }
  double ymax = 0.0;
  for (const double v : y) ymax = std::fmax(ymax, v);
  if (!(y0 > 0.0) || !(ymax > rise * y0)) return none;
  const double lo = rise * y0;
  const double hi = ceiling * ymax;
  if (!(hi > lo)) return none;
  double t0 = std::nan(""), t1 = std::nan("");
  for (std::size_t i = 0; i < t.size() && i < y.size(); ++i) {
    if (!std::isfinite(t0) && y[i] >= lo) t0 = t[i];
    if (std::isfinite(t0) && y[i] <= hi) t1 = t[i];
  }
  if (!std::isfinite(t0) || !std::isfinite(t1) || !(t1 > t0)) return none;
  return {t0, t1};
}

/**
 * @brief Oscillation frequency from the spacing of successive minima of a
 *        damped oscillation.
 *
 * The electric-field energy of a Landau-damped wave has minima every half
 * period, so the mean spacing of consecutive minima is `pi/omega`. Using
 * minima rather than maxima is deliberate: the minima of `|E|^2` are sharp
 * and deep, while the maxima sit on a slowly varying envelope and their
 * positions are much less well determined.
 */
[[nodiscard]] inline double frequency_from_minima(const std::vector<double> &t,
                                                  const std::vector<double> &y,
                                                  double t0, double t1) {
  std::vector<double> mins;
  for (std::size_t i = 1; i + 1 < t.size() && i + 1 < y.size(); ++i) {
    if (t[i] < t0 || t[i] > t1) continue;
    if (y[i] < y[i - 1] && y[i] < y[i + 1]) {
      // Parabolic refinement through the three samples: the sample grid is
      // far coarser than the precision wanted on the period.
      const double a = y[i - 1], b = y[i], c = y[i + 1];
      const double den = a - 2.0 * b + c;
      const double sh = (std::fabs(den) > 0.0) ? 0.5 * (a - c) / den : 0.0;
      const double dt = 0.5 * (t[i + 1] - t[i - 1]);
      mins.push_back(t[i] + sh * dt);
    }
  }
  if (mins.size() < 2) {
    return std::nan("");
  }
  double sum = 0.0;
  for (std::size_t i = 1; i < mins.size(); ++i) {
    sum += mins[i] - mins[i - 1];
  }
  const double half_period = sum / static_cast<double>(mins.size() - 1);
  return std::acos(-1.0) / half_period;
}

/**
 * @brief Discrete-velocity recurrence time \(T_R = 2\pi / (k\,\Delta v)\).
 *
 * Free-streaming a spatially periodic mode on a uniform velocity grid
 * reconstructs the initial density perturbation when neighbouring cells
 * differ in phase by \(2\pi\). That is a property of the grid, not of
 * Landau damping, and a science fit that includes \(t \gtrsim T_R\) is
 * measuring the recycled initial condition.
 *
 * Cell-centred \(v_j = -v_{\max} + (j + \tfrac12)\Delta v\) puts a global
 * minus sign on the *signed* mode at \(t = T_R\); the observable is the
 * modulus. See `test_recurrence.cpp`.
 */
[[nodiscard]] inline double recurrence_time(double k, double dv) {
  if (!(k > 0.0) || !(dv > 0.0) || !std::isfinite(k) || !std::isfinite(dv)) {
    throw std::invalid_argument(
        "recurrence_time: need k > 0 and dv > 0, got k = " +
        std::to_string(k) + ", dv = " + std::to_string(dv));
  }
  return 2.0 * std::acos(-1.0) / (k * dv);
}

/**
 * @brief Refuse a science-rate window that has walked into recurrence.
 *
 * A fit that includes \(t > 0.8\,T_R\) is not a damping or growth rate;
 * the discrete velocity grid is reconstructing the seed. The driver must
 * fail loudly rather than auto-extend `t_end` or quote the recycled
 * amplitude. @p t1 is the end of the fit window, not of the run.
 */
inline void require_fit_before_recurrence(double t1, double k, double dv,
                                          double frac = 0.8) {
  if (!std::isfinite(t1)) {
    return;
  }
  const double tr = recurrence_time(k, dv);
  if (t1 > frac * tr) {
    throw std::runtime_error(
        "science fit window ends at t = " + std::to_string(t1) +
        " which is past " + std::to_string(frac) +
        " of the recurrence time T_R = " + std::to_string(tr) +
        " (k = " + std::to_string(k) + ", dv = " + std::to_string(dv) +
        "). Shorten t_end or refine the velocity grid; the driver will not "
        "extend the run past T_R.");
  }
}

/// How @ref find_recurrence_peak failed, when it did. Empty is not "no
/// revival": an all-zero series has nothing to recur, while a decaying
/// mode that never comes back is a real signal in the wrong regime.
enum class RevivalKind { empty, none, found };

/// One measured recurrence peak of a strictly positive amplitude series.
struct Revival {
  RevivalKind kind{RevivalKind::empty};
  double t{std::nan("")};
  double amplitude{0.0};
};

/**
 * @brief Time of the recurrence peak of \(|\hat\rho(k)|\) (or any positive
 *        amplitude series that is supposed to revive at @p t_pred).
 *
 * Searches \([0.5, 1.3]\,t_{\mathrm{pred}}\) for a local maximum of
 * amplitude at least @p floor_frac of the first positive sample, then
 * refines with the same three-point parabola as
 * @ref frequency_from_minima. Returns `empty` when there is no signal to
 * measure, `none` when there is a signal but no revival, and `found`
 * with the refined time otherwise. Those three are distinct on purpose:
 * treating "the reduction was zero" as "recurrence was late" would pass
 * a broken deposition.
 */
[[nodiscard]] inline Revival
find_recurrence_peak(const std::vector<double> &t, const std::vector<double> &y,
                     double t_pred, double floor_frac = 0.3) {
  Revival r;
  if (t.size() < 3 || y.size() < 3 || !(t_pred > 0.0) || !std::isfinite(t_pred)) {
    r.kind = RevivalKind::empty;
    return r;
  }
  double y0 = 0.0;
  for (const double v : y) {
    if (v > 0.0 && std::isfinite(v)) {
      y0 = v;
      break;
    }
  }
  if (!(y0 > 0.0)) {
    r.kind = RevivalKind::empty;
    return r;
  }
  const double t_lo = 0.5 * t_pred;
  const double t_hi = 1.3 * t_pred;
  std::size_t imax = t.size();
  double ymax = 0.0;
  for (std::size_t i = 1; i + 1 < t.size() && i + 1 < y.size(); ++i) {
    if (t[i] < t_lo || t[i] > t_hi) continue;
    if (!(y[i] > 0.0) || !std::isfinite(y[i])) continue;
    if (y[i] > ymax) {
      ymax = y[i];
      imax = i;
    }
  }
  if (imax >= t.size() || ymax < floor_frac * y0) {
    r.kind = RevivalKind::none;
    return r;
  }
  if (!(y[imax] >= y[imax - 1] && y[imax] >= y[imax + 1])) {
    r.kind = RevivalKind::none;
    return r;
  }
  const double a = y[imax - 1];
  const double b = y[imax];
  const double c = y[imax + 1];
  const double den = a - 2.0 * b + c;
  const double sh = (std::fabs(den) > 0.0) ? 0.5 * (a - c) / den : 0.0;
  const double dt = 0.5 * (t[imax + 1] - t[imax - 1]);
  r.kind = RevivalKind::found;
  r.t = t[imax] + sh * dt;
  r.amplitude = ymax;
  return r;
}

/**
 * @brief Rank-0 append-only CSV sink.
 *
 * Opens in append mode and writes the header only when the file is empty, so
 * re-running a driver adds rows instead of destroying the previous run's
 * measurements. That is deliberate: a resolution study is a sequence of runs
 * whose output belongs in one file, and an accidental clobber of a run that
 * took ten minutes is a worse failure mode than a file with two headers'
 * worth of history in it. Each row carries the caller's `run_id`, so rows
 * from different runs stay separable.
 *
 * Every method is a no-op on ranks other than 0.
 */
class CsvAppender {
public:
  CsvAppender() = default;

  CsvAppender(const std::string &path, const std::string &header, int rank)
      : m_rank(rank) {
    if (rank != 0 || path.empty()) {
      return;
    }
    const std::filesystem::path fp(path);
    if (fp.has_parent_path() && !fp.parent_path().empty()) {
      std::filesystem::create_directories(fp.parent_path());
    }
    std::error_code ec;
    const auto sz = std::filesystem::file_size(fp, ec);
    const bool fresh = ec || sz == 0;
    m_os.open(path, std::ios::out | std::ios::app);
    if (!m_os) {
      throw std::runtime_error("alloy_dendrite: cannot open CSV for append: " +
                               path);
    }
    if (fresh) {
      m_os << header << "\n";
    }
    m_open = true;
  }

  [[nodiscard]] bool active() const noexcept { return m_open && m_rank == 0; }

  /// Append one preformatted row (no trailing newline needed) and flush, so
  /// a run killed mid-way still leaves every completed sample on disk.
  void row(const std::string &line) {
    if (!active()) {
      return;
    }
    m_os << line << "\n";
    m_os.flush();
  }

private:
  std::ofstream m_os;
  bool m_open{false};
  int m_rank{0};
};

/**
 * @brief Rotation rate of a planar vector series, by unwrapped phase.
 *
 * The gyro-motion oracle. Under a uniform `B_z` and no self-consistent
 * field the Lorentz force does no work, so the mean velocity
 * `(<v_x>, <v_y>)` rotates rigidly at `omega_c = sigma B_z / mu`. Fitting
 * the *unwrapped* phase rather than counting periods is what lets the
 * measurement use every sample and resolve a rate to well below one
 * sample per period.
 *
 * Unwrapping is done incrementally, which requires the samples to be
 * closer together than half a rotation; that is checked, because a silent
 * wrap would report a rate slower by exactly `2 pi / dt_sample` and look
 * entirely plausible.
 */
[[nodiscard]] inline double fit_rotation_rate(const std::vector<double> &t,
                                              const std::vector<double> &px,
                                              const std::vector<double> &py,
                                              bool *aliased = nullptr) {
  const double pi = std::acos(-1.0);
  std::vector<double> ph;
  ph.reserve(t.size());
  double unwrapped = 0.0;
  double prev = 0.0;
  bool wrapped = false;
  for (std::size_t i = 0; i < t.size() && i < px.size() && i < py.size(); ++i) {
    const double a = std::atan2(py[i], px[i]);
    if (i == 0) {
      unwrapped = a;
    } else {
      // Principal value of the increment. Reducing `a - prev` into
      // `(-pi, pi]` is the unwrapping; the flag is about whether that
      // reduction was a *choice*. An earlier revision instead flagged the
      // raw `a - prev` whenever it exceeded 1.5 pi, which is what a
      // perfectly ordinary atan2 branch cut looks like -- so it reported
      // aliasing on a rotation sampled forty times per turn. The test
      // caught it; the estimator is the apparatus, so it gets tested like
      // one.
      double d = a - prev;
      d -= 2.0 * pi * std::round(d / (2.0 * pi));
      if (std::fabs(d) > 0.9 * pi) wrapped = true;
      unwrapped += d;
    }
    prev = a;
    ph.push_back(unwrapped);
  }
  if (aliased != nullptr) *aliased = wrapped;
  const std::size_t n = ph.size();
  if (n < 3) return std::nan("");
  double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sx += t[i];
    sy += ph[i];
    sxx += t[i] * t[i];
    sxy += t[i] * ph[i];
  }
  const double dn = static_cast<double>(n);
  const double den = dn * sxx - sx * sx;
  if (std::fabs(den) < 1.0e-300) return std::nan("");
  return (dn * sxy - sx * sy) / den;
}

} // namespace vlasov
