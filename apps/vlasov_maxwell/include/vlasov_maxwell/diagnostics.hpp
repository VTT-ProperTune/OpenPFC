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
#include <cmath>
#include <cstdio>
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

/// Relative change of @p now from @p ref, or the absolute change when @p ref
/// is too small for a ratio to mean anything.
[[nodiscard]] inline double drift(double now, double ref,
                                  double floor = 1.0e-300) noexcept {
  const double a = std::fabs(ref);
  return (a > floor) ? (now - ref) / a : (now - ref);
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
            int step) {
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
  const std::vector<double> zero(fields.Ex.size(), 0.0);
  L.energy_ex = field_energy(line, fields.Ex);
  L.energy_bz = field_energy(line, fields.Bz);
  L.energy_em = transverse_energy(line, fields.Ey, fields.Bz);
  L.field_momentum_x = field_momentum_x(line, fields.Ey, fields.Bz);
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
         "energy_ex,energy_em,energy_bz,field_momentum_x,"
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
      "%.17g,%.17g,%.17g,%.17g,"
      "%.17g,%.17g,"
      "%.6e,%.6e,%.6e,"
      "%.6e,%.6e,"
      "%.6e,%.6e,%.6e,%.6e,%.6e,%.6e",
      run_id.c_str(), L.step, L.t, L.number, L.kinetic_energy, L.momentum_x,
      L.momentum_y, L.l1, L.l2, L.entropy, L.f_min, L.f_max, L.energy_ex,
      L.energy_em, L.energy_bz, L.field_momentum_x, L.total_energy,
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

} // namespace vlasov
