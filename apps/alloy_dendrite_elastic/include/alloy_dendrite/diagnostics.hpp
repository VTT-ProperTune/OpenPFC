// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file diagnostics.hpp
 * @brief First-class measurements for the thermo-solutal alloy core:
 *        conservation, planar-front kinetics, effective partition
 *        coefficient, and 2-D dendrite tip metrics, all written to CSV.
 *
 * @details
 * ## Why the definitions live here and are this pedantic
 *
 * "Tip velocity" and "tip radius" are not quantities a grid has; they are
 * quantities a *measurement procedure* has. Two people with the same field
 * and different procedures get answers that differ by more than the physics
 * they are trying to resolve -- a staircased `phi = 0` contour makes a
 * finite-difference tip velocity oscillate by tens of percent while the
 * underlying front is perfectly smooth. So every number this header produces
 * is defined operationally, in one place, and the procedure is part of the
 * output rather than folklore. A later agent doing science runs should be
 * able to read this file and know exactly what it is plotting.
 *
 * ## What is measured
 *
 * | Quantity | Definition |
 * |---|---|
 * | `solute_total` | `sum_cells [ P(phi)/(1-k) + psi ]`, i.e. `sum c / (c_l^0
 * (1-k))`. Invariant to round-off on a periodic grid; see `step.hpp`. | |
 * `heat_balance` | `sum_cells theta - (1/2) sum_cells phi`. Invariant to round-off
 * whenever equation (4) is integrated on a periodic grid, because the thermal
 * Laplacian telescopes to zero and the only source is `(1/2) d_t phi`. | | `x_if` |
 * Interface position: the largest `x` above the slab centre where the
 * transverse-mean `phi` profile crosses zero downward, by linear interpolation
 * between the bracketing cells. | | `V` | Least-squares slope of `x_if(t)` over the
 * trailing fraction of samples. Not a finite difference: a staircased crossing is a
 * high-frequency signal and differencing amplifies it (PR #104 reached the same
 * conclusion for its tip velocity). | | `ell` | Solute boundary-layer width from a
 * log-linear fit of `U - U_far` against `x` in the liquid ahead of the front.
 * Compare with `D_l / V`. | | `U_i` | That same exponential fit extrapolated back to
 * `x_if`. This is the *outer* solution at the interface -- the quantity the
 * thin-interface Gibbs-Thomson relation `U_i = -beta V` refers to, not the raw field
 * value inside the diffuse profile. | | `U_s` | Mean `U` over freshly formed solid,
 * `phi > 0.99` in a band `[x_if - 15 W0, x_if - 5 W0]`. | | `k_eff` | `k (1 + (1-k)
 * U_s) / (1 + (1-k) U_i)`, which is `c_s / c_l^i` exactly. Equals `k` iff `U_s =
 * U_i`. | | `x_tip`, `rho_tip` | 2-D dendrite; see @ref measure_tip. |
 *
 * ## Why `k_eff` is defined through `U_i` and not through a nearby cell
 *
 * The obvious implementation -- walk outward from the interface, take the
 * first cell with `phi < -0.9` as `c_l` -- samples a point that is still
 * inside the diffuse profile and whose distance from the interface depends
 * on `dx` and on `W0`. The measured `k_eff` then carries a resolution
 * artefact of the same size as the effect being measured. Extrapolating the
 * outer exponential back to the interface removes that: it is the same
 * matching the asymptotic analysis performs, so it compares like with like.
 *
 * @see step.hpp for why the conserved quantities are conserved
 * @see parameters.hpp for the predictions these measurements are tested against
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/grid_field.hpp>

#include <alloy_dendrite/parameters.hpp>

namespace alloy_dendrite {

using HostField = pfc::data::Field<double, pfc::HostSpace>;

/// Globally reduced invariants of equations (3) and (4).
struct Conservation {
  /// `sum c / (c_l^0 (1-k))`. Conserved exactly on a periodic grid.
  double solute_total{0.0};
  /// `sum theta - (1/2) sum phi`. Conserved exactly when (4) is integrated.
  double heat_balance{0.0};
  double phi_total{0.0};
  double theta_total{0.0};
  double u_min{0.0};
  double u_max{0.0};
  double phi_min{0.0};
  double phi_max{0.0};
};

/// Reduce the conserved quantities and the field ranges across all ranks.
[[nodiscard]] inline Conservation
measure_conservation(const HostField &phi, const HostField &U,
                     const HostField &theta, const ModelParams &p, MPI_Comm comm) {
  const double k = p.k;
  const double inv_1mk = 1.0 / (1.0 - k);
  double s_sol = 0.0;
  double s_phi = 0.0;
  double s_th = 0.0;
  double umin = std::numeric_limits<double>::infinity();
  double umax = -std::numeric_limits<double>::infinity();
  double pmin = std::numeric_limits<double>::infinity();
  double pmax = -std::numeric_limits<double>::infinity();
  phi.for_each_owned([&](int i, int j, int kk) {
    const double ph = phi(i, j, kk);
    const double uu = U(i, j, kk);
    const double pp = solute_prefactor(k, ph);
    s_sol += pp * inv_1mk + pp * uu;
    s_phi += ph;
    s_th += theta(i, j, kk);
    umin = std::fmin(umin, uu);
    umax = std::fmax(umax, uu);
    pmin = std::fmin(pmin, ph);
    pmax = std::fmax(pmax, ph);
  });
  double loc_sum[3] = {s_sol, s_phi, s_th};
  double glob_sum[3] = {0.0, 0.0, 0.0};
  MPI_Allreduce(loc_sum, glob_sum, 3, MPI_DOUBLE, MPI_SUM, comm);
  double loc_min[2] = {umin, pmin};
  double glob_min[2] = {0.0, 0.0};
  MPI_Allreduce(loc_min, glob_min, 2, MPI_DOUBLE, MPI_MIN, comm);
  double loc_max[2] = {umax, pmax};
  double glob_max[2] = {0.0, 0.0};
  MPI_Allreduce(loc_max, glob_max, 2, MPI_DOUBLE, MPI_MAX, comm);

  Conservation c;
  c.solute_total = glob_sum[0];
  c.phi_total = glob_sum[1];
  c.theta_total = glob_sum[2];
  c.heat_balance = c.theta_total - 0.5 * c.phi_total;
  c.u_min = glob_min[0];
  c.phi_min = glob_min[1];
  c.u_max = glob_max[0];
  c.phi_max = glob_max[1];
  return c;
}

/**
 * @brief Whether the final field is still a phase field.
 *
 * `phi` in `[-1.1, 1.1]` (a fourth-order stencil overshoots a tanh front by
 * ~1e-4, so 0.1 of headroom) and `U` finite. A run that left this band has
 * diverged; see @ref dendrite_result_valid for why that matters.
 */
[[nodiscard]] inline bool conservation_state_finite(const Conservation &c) noexcept {
  return std::isfinite(c.phi_min) && std::isfinite(c.phi_max) &&
         std::isfinite(c.u_min) && std::isfinite(c.u_max) && c.phi_min > -1.1 &&
         c.phi_max < 1.1;
}

/**
 * @brief Whether a dendrite run's headline numbers may be quoted.
 *
 * A finite `v_tip` is not enough. Failed tip samples are skipped, so a
 * trailing-window fit of the remaining crossings still reports a plausible
 * velocity for a field that is full of NaN -- measured, `v_tip = 0.065` on a
 * run that blew up at `t = 1065` of `t_end = 2000`. Validity is therefore
 * the conjunction of a finite slope, a still-bounded final state, and a
 * successful measurement on the *last* sample (early seed samples may fail;
 * a failure at the end is a divergence).
 */
[[nodiscard]] inline bool dendrite_result_valid(double v_tip, bool state_finite,
                                                bool last_sample_valid) noexcept {
  return std::isfinite(v_tip) && state_finite && last_sample_valid;
}

/**
 * @brief Transverse-mean profile of a field along `x`, valid on every rank.
 *
 * Implemented as a length-`Nx` `MPI_Allreduce` rather than a gather, so it is
 * independent of how the domain happens to be decomposed (slabs along any
 * axis, or pencils) and costs `O(Nx)` regardless of rank count.
 */
[[nodiscard]] inline std::vector<double> transverse_mean_profile(const HostField &f,
                                                                 MPI_Comm comm) {
  const auto gsz = f.global_size();
  const int nxg = gsz[0];
  std::vector<double> local(static_cast<std::size_t>(nxg), 0.0);
  f.for_each_owned([&](int i, int j, int kk) {
    local[static_cast<std::size_t>(f.lower_global()[0] + i)] += f(i, j, kk);
  });
  std::vector<double> global(static_cast<std::size_t>(nxg), 0.0);
  MPI_Allreduce(local.data(), global.data(), nxg, MPI_DOUBLE, MPI_SUM, comm);
  const double inv =
      1.0 / (static_cast<double>(gsz[1]) * static_cast<double>(gsz[2]));
  for (auto &v : global) {
    v *= inv;
  }
  return global;
}

/**
 * @brief Global `Nx * Ny` slice at global `k = k_plane`, on every rank.
 *
 * Same trick as @ref transverse_mean_profile and for the same reason: an
 * `MPI_Allreduce` over the plane is independent of how the domain is
 * decomposed, whereas a `Gatherv` has to know. It costs `Nx * Ny` doubles
 * per call, which for the diagnostic cadence used here is nothing next to
 * the step itself, and it makes the 2-D and 3-D tip measurements literally
 * the same code with a different `k_plane`.
 *
 * Cells outside the plane contribute zero, so the result is exact rather
 * than an average.
 */
[[nodiscard]] inline std::vector<double>
global_xy_plane(const HostField &f, int k_plane, MPI_Comm comm) {
  const auto gsz = f.global_size();
  const std::size_t n =
      static_cast<std::size_t>(gsz[0]) * static_cast<std::size_t>(gsz[1]);
  std::vector<double> local(n, 0.0);
  const auto lo = f.lower_global();
  f.for_each_owned([&](int i, int j, int kk) {
    if (lo[2] + kk != k_plane) {
      return;
    }
    local[static_cast<std::size_t>(lo[0] + i) +
          static_cast<std::size_t>(lo[1] + j) * static_cast<std::size_t>(gsz[0])] =
        f(i, j, kk);
  });
  std::vector<double> global(n, 0.0);
  MPI_Allreduce(local.data(), global.data(), static_cast<int>(n), MPI_DOUBLE,
                MPI_SUM, comm);
  return global;
}

/// Result of @ref measure_planar_front. `valid` is false when the profile did
/// not contain a usable front or the exponential fit had too few points; the
/// drivers then write the row with NaNs rather than a fabricated number.
struct PlanarFront {
  bool valid{false};
  double x_if{std::numeric_limits<double>::quiet_NaN()};
  double u_far{std::numeric_limits<double>::quiet_NaN()};
  double u_interface{std::numeric_limits<double>::quiet_NaN()};
  double u_solid{std::numeric_limits<double>::quiet_NaN()};
  double ell{std::numeric_limits<double>::quiet_NaN()};
  double k_eff{std::numeric_limits<double>::quiet_NaN()};
  /// Coefficient of determination of the log-linear boundary-layer fit.
  double fit_r2{std::numeric_limits<double>::quiet_NaN()};
  int fit_points{0};
};

/// Windows, in units of `W0`, that define the planar measurement. Exposed as
/// a struct so a driver can widen them for a coarse grid and so the values
/// used end up in the CSV instead of only in this comment.
struct PlanarWindows {
  /// Distance ahead of `x_if` where the exponential fit starts. Must clear
  /// the diffuse profile; `tanh` is within 1e-4 of `-1` by `5 W0 / sqrt(2)`.
  double fit_skip = 5.0;
  /// Fit stops once `U - U_far` has decayed to this fraction of its value at
  /// the start of the window, or at the far point, whichever comes first.
  double fit_decay_floor = 0.02;
  /// Band behind `x_if`, in `W0`, over which `U_solid` is averaged.
  double solid_lo = 5.0;
  double solid_hi = 15.0;
  /// `phi` above which a cell counts as solid for `U_solid`.
  double solid_phi = 0.99;
};

/**
 * @brief Measure the right-hand front of a periodic two-front planar run.
 *
 * @param phi_prof  Transverse-mean `phi` profile, length `Nx`.
 * @param u_prof    Transverse-mean `U` profile, length `Nx`.
 * @param dx        Grid spacing.
 * @param p         Model parameters (`k` and `W0` are used).
 * @param win       Measurement windows.
 *
 * The initial condition is a solid slab centred in a periodic box, so the
 * profile is `+1` in the middle and `-1` at both ends and there are exactly
 * two fronts. This routine measures the one moving in `+x`: it scans upward
 * from the slab centre for the first downward zero crossing.
 */
[[nodiscard]] inline PlanarFront
measure_planar_front(const std::vector<double> &phi_prof,
                     const std::vector<double> &u_prof, double dx,
                     const ModelParams &p, const PlanarWindows &win = {}) {
  PlanarFront out;
  const int n = static_cast<int>(phi_prof.size());
  if (n < 16 || u_prof.size() != phi_prof.size()) {
    return out;
  }
  const int centre = n / 2;

  int i_cross = -1;
  for (int i = centre; i + 1 < n; ++i) {
    if (phi_prof[static_cast<std::size_t>(i)] >= 0.0 &&
        phi_prof[static_cast<std::size_t>(i + 1)] < 0.0) {
      i_cross = i;
      break;
    }
  }
  if (i_cross < 0) {
    return out;
  }
  const double p0 = phi_prof[static_cast<std::size_t>(i_cross)];
  const double p1 = phi_prof[static_cast<std::size_t>(i_cross + 1)];
  const double frac = p0 / (p0 - p1); // p0 >= 0 > p1, so frac is in [0, 1)
  out.x_if = (static_cast<double>(i_cross) + frac) * dx;

  // The far point is the cell farthest from both fronts. The slab is centred
  // at `n/2`, so the liquid wraps through the periodic seam and its midpoint
  // is index 0. The fit therefore runs from just ahead of the right front up
  // to the last cell before the seam; there is no need to walk across it.
  out.u_far = u_prof[0];
  const int i_limit = n - 1;

  // ---- exponential boundary layer -------------------------------------
  const int i_start =
      i_cross + 1 + static_cast<int>(std::ceil(win.fit_skip * p.W0 / dx));
  if (i_start >= i_limit - 8) {
    return out;
  }
  const double d_start = u_prof[static_cast<std::size_t>(i_start)] - out.u_far;
  if (!(d_start > 0.0)) {
    return out;
  }
  const double d_stop = win.fit_decay_floor * d_start;
  std::vector<double> xs;
  std::vector<double> ys;
  for (int i = i_start; i <= i_limit; ++i) {
    const double d = u_prof[static_cast<std::size_t>(i)] - out.u_far;
    if (!(d > d_stop)) {
      break;
    }
    xs.push_back(static_cast<double>(i) * dx - out.x_if);
    ys.push_back(std::log(d));
  }
  out.fit_points = static_cast<int>(xs.size());
  if (out.fit_points < 8) {
    return out;
  }
  const double m = static_cast<double>(out.fit_points);
  double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0, syy = 0.0;
  for (std::size_t q = 0; q < xs.size(); ++q) {
    sx += xs[q];
    sy += ys[q];
    sxx += xs[q] * xs[q];
    sxy += xs[q] * ys[q];
    syy += ys[q] * ys[q];
  }
  const double den = m * sxx - sx * sx;
  if (!(std::fabs(den) > 0.0)) {
    return out;
  }
  const double slope = (m * sxy - sx * sy) / den;
  const double icept = (sy - slope * sx) / m;
  if (!(slope < 0.0)) {
    return out;
  }
  out.ell = -1.0 / slope;
  // Extrapolate the outer solution back to the interface: at x = x_if the
  // fit variable is 0, so U_i = U_far + exp(intercept).
  out.u_interface = out.u_far + std::exp(icept);
  const double sst = syy - sy * sy / m;
  const double ssr = slope * slope * (sxx - sx * sx / m);
  out.fit_r2 = (sst > 0.0) ? (ssr / sst) : std::numeric_limits<double>::quiet_NaN();

  // ---- freshly formed solid -------------------------------------------
  const int j_hi = i_cross - static_cast<int>(std::floor(win.solid_lo * p.W0 / dx));
  const int j_lo = i_cross - static_cast<int>(std::ceil(win.solid_hi * p.W0 / dx));
  double acc = 0.0;
  int cnt = 0;
  for (int i = std::max(0, j_lo); i <= std::min(n - 1, j_hi); ++i) {
    if (phi_prof[static_cast<std::size_t>(i)] > win.solid_phi) {
      acc += u_prof[static_cast<std::size_t>(i)];
      ++cnt;
    }
  }
  if (cnt == 0) {
    return out;
  }
  out.u_solid = acc / static_cast<double>(cnt);

  const double k = p.k;
  out.k_eff =
      k * (1.0 + (1.0 - k) * out.u_solid) / (1.0 + (1.0 - k) * out.u_interface);
  out.valid = true;
  return out;
}

/**
 * @brief Least-squares slope of `y` against `x` over the trailing
 *        @p fraction of the samples.
 *
 * Used for the front and tip velocities. A first difference of a level-set
 * crossing is dominated by grid pinning -- the crossing snaps between cells
 * -- and higher-order differences make it worse, not better, because the
 * staircase is a high-frequency signal. A least-squares slope over many
 * pinning cycles is the estimator that converges.
 */
[[nodiscard]] inline double trailing_slope(const std::vector<double> &x,
                                           const std::vector<double> &y,
                                           double fraction) {
  const std::size_t n = std::min(x.size(), y.size());
  if (n < 4) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const std::size_t want = std::max<std::size_t>(
      4, static_cast<std::size_t>(fraction * static_cast<double>(n)));
  const std::size_t begin = (want >= n) ? 0 : (n - want);
  double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
  double m = 0.0;
  for (std::size_t q = begin; q < n; ++q) {
    if (!std::isfinite(y[q])) {
      continue;
    }
    sx += x[q];
    sy += y[q];
    sxx += x[q] * x[q];
    sxy += x[q] * y[q];
    m += 1.0;
  }
  if (m < 4.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double den = m * sxx - sx * sx;
  if (!(std::fabs(den) > 0.0)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return (m * sxy - sx * sy) / den;
}

/// Result of @ref measure_tip. Distances are in the same units as `dx`.
struct DendriteTip {
  bool valid{false};
  double x_tip{std::numeric_limits<double>::quiet_NaN()};
  double y_tip{std::numeric_limits<double>::quiet_NaN()};
  /// Radius of curvature from the parabola fit; `+inf` for a flat front.
  double rho{std::numeric_limits<double>::quiet_NaN()};
  /// RMS residual of the parabola fit, in units of `dx`. A large value means
  /// the contour near the tip is not parabolic and `rho` should not be
  /// trusted -- reported so that judgement is possible.
  double fit_rms{std::numeric_limits<double>::quiet_NaN()};
  int fit_rows{0};
};

/**
 * @brief 2-D tip position and radius from a rank-0 gathered `phi` slab.
 *
 * @param phi_xy  Row-major `Nx * Ny` slab (x fastest), rank 0 only.
 * @param nx,ny   Global extents.
 * @param dx,dy   Grid spacings.
 * @param i_seed,j_seed  Seed cell; the `+x` arm is measured.
 * @param half_width_cells  Rows either side of the tip row used for the
 *                          parabola fit.
 *
 * **Tip position.** For each row `j` in the band, the crossing `x_c(j)` is
 * the largest `x > x_seed` at which `phi` changes sign from `+` to `-`,
 * linearly interpolated. The tip row `j_tip` is the row with the largest
 * `x_c`, and `x_tip = x_c(j_tip)`. Taking the maximum over a band rather
 * than trusting the seed row means a run that loses its symmetry still
 * reports the actual tip, and a run that keeps it (which a deterministic
 * 4-fold case does) reports the seed row, deterministically.
 *
 * **Tip radius.** Near the tip the `phi = 0` contour of a dendrite is
 * parabolic, `x(y) = x_tip - (y - y_tip)^2 / (2 rho)`. Fitting `x_c(j)`
 * against `(y_j - y_tip)^2` by ordinary least squares gives an intercept and
 * a slope `-1/(2 rho)`; `rho` is minus the reciprocal of twice that slope.
 * The fit uses `2 * half_width_cells + 1` rows centred on `j_tip`. The
 * half-width is a *parameter*, not a constant, because the parabolic
 * description is only good within roughly `rho` of the tip: too narrow and
 * the fit is dominated by the staircase, too wide and it is biased by the
 * non-parabolic flanks. @ref DendriteTip::fit_rms is reported so that the
 * choice can be checked rather than assumed.
 */
[[nodiscard]] inline DendriteTip measure_tip(const std::vector<double> &phi_xy,
                                             int nx, int ny, double dx, double dy,
                                             int i_seed, int j_seed,
                                             int half_width_cells) {
  DendriteTip out;
  if (static_cast<int>(phi_xy.size()) != nx * ny || nx < 8 || ny < 8) {
    return out;
  }
  auto at = [&](int i, int j) {
    return phi_xy[static_cast<std::size_t>(i) +
                  static_cast<std::size_t>(j) * static_cast<std::size_t>(nx)];
  };
  auto crossing = [&](int j) -> double {
    double best = std::numeric_limits<double>::quiet_NaN();
    for (int i = i_seed; i + 1 < nx; ++i) {
      const double a = at(i, j);
      const double b = at(i + 1, j);
      if (a >= 0.0 && b < 0.0) {
        best = (static_cast<double>(i) + a / (a - b)) * dx;
      }
    }
    return best;
  };

  const int band = std::max(half_width_cells, 2);
  int j_tip = -1;
  double x_best = -std::numeric_limits<double>::infinity();
  for (int j = j_seed - band; j <= j_seed + band; ++j) {
    if (j < 0 || j >= ny) {
      continue;
    }
    const double xc = crossing(j);
    if (std::isfinite(xc) && xc > x_best) {
      x_best = xc;
      j_tip = j;
    }
  }
  if (j_tip < 0) {
    return out;
  }
  out.x_tip = x_best;
  out.y_tip = static_cast<double>(j_tip) * dy;

  std::vector<double> yy;
  std::vector<double> xc;
  for (int j = j_tip - half_width_cells; j <= j_tip + half_width_cells; ++j) {
    if (j < 0 || j >= ny) {
      continue;
    }
    const double c = crossing(j);
    if (!std::isfinite(c)) {
      continue;
    }
    const double dyj = (static_cast<double>(j) - static_cast<double>(j_tip)) * dy;
    yy.push_back(dyj * dyj);
    xc.push_back(c);
  }
  out.fit_rows = static_cast<int>(yy.size());
  if (out.fit_rows < 3) {
    return out;
  }
  const double m = static_cast<double>(out.fit_rows);
  double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
  for (std::size_t q = 0; q < yy.size(); ++q) {
    sx += yy[q];
    sy += xc[q];
    sxx += yy[q] * yy[q];
    sxy += yy[q] * xc[q];
  }
  const double den = m * sxx - sx * sx;
  if (!(std::fabs(den) > 0.0)) {
    return out;
  }
  const double slope = (m * sxy - sx * sy) / den;
  const double icept = (sy - slope * sx) / m;
  double ss = 0.0;
  for (std::size_t q = 0; q < yy.size(); ++q) {
    const double r = xc[q] - (icept + slope * yy[q]);
    ss += r * r;
  }
  out.fit_rms = std::sqrt(ss / m) / dx;
  out.rho = (slope < 0.0) ? (-0.5 / slope) : std::numeric_limits<double>::infinity();
  out.valid = true;
  return out;
}

/**
 * @brief Most-downstream solid-to-liquid crossing on a gathered `xy` plane.
 *
 * @ref measure_tip is the `+x` arm of a seed at the box centre: it only
 * searches a band of rows around `j_seed`, starting at `i_seed`. FTA
 * directional solidification grows from a cold wall, so the tip that
 * matters is the globally most-downstream (`largest x`) `+` to `-`
 * crossing, not that arm. @p i_start is the first cell that may be solid;
 * @p j_lo / @p j_hi (inclusive) clip the search. The parabola fit is the
 * same as @ref measure_tip once the tip row is known.
 */
[[nodiscard]] inline DendriteTip
measure_downstream_tip(const std::vector<double> &phi_xy, int nx, int ny,
                       double dx, double dy, int i_start, int j_lo, int j_hi,
                       int half_width_cells) {
  DendriteTip out;
  if (static_cast<int>(phi_xy.size()) != nx * ny || nx < 8 || ny < 2) {
    return out;
  }
  auto at = [&](int i, int j) {
    return phi_xy[static_cast<std::size_t>(i) +
                  static_cast<std::size_t>(j) * static_cast<std::size_t>(nx)];
  };
  auto crossing = [&](int j) -> double {
    double best = std::numeric_limits<double>::quiet_NaN();
    const int i0 = std::max(0, i_start);
    for (int i = i0; i + 1 < nx; ++i) {
      const double a = at(i, j);
      const double b = at(i + 1, j);
      if (a >= 0.0 && b < 0.0) {
        best = (static_cast<double>(i) + a / (a - b)) * dx;
      }
    }
    return best;
  };

  const int ja = std::max(0, std::min(j_lo, j_hi));
  const int jb = std::min(ny - 1, std::max(j_lo, j_hi));
  int j_tip = -1;
  double x_best = -std::numeric_limits<double>::infinity();
  for (int j = ja; j <= jb; ++j) {
    const double xc = crossing(j);
    if (std::isfinite(xc) && xc > x_best) {
      x_best = xc;
      j_tip = j;
    }
  }
  if (j_tip < 0) {
    return out;
  }
  out.x_tip = x_best;
  out.y_tip = static_cast<double>(j_tip) * dy;

  std::vector<double> yy;
  std::vector<double> xc;
  for (int j = j_tip - half_width_cells; j <= j_tip + half_width_cells; ++j) {
    if (j < 0 || j >= ny) {
      continue;
    }
    const double c = crossing(j);
    if (!std::isfinite(c)) {
      continue;
    }
    const double dyj = (static_cast<double>(j) - static_cast<double>(j_tip)) * dy;
    yy.push_back(dyj * dyj);
    xc.push_back(c);
  }
  out.fit_rows = static_cast<int>(yy.size());
  if (out.fit_rows < 3) {
    // Position is still a measurement even if the parabola fit has too few
    // rows -- FTA cells and grooves are often that narrow.
    out.valid = std::isfinite(out.x_tip);
    return out;
  }
  const double m = static_cast<double>(out.fit_rows);
  double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
  for (std::size_t q = 0; q < yy.size(); ++q) {
    sx += yy[q];
    sy += xc[q];
    sxx += yy[q] * yy[q];
    sxy += yy[q] * xc[q];
  }
  const double den = m * sxx - sx * sx;
  if (!(std::fabs(den) > 0.0)) {
    out.valid = true;
    return out;
  }
  const double slope = (m * sxy - sx * sy) / den;
  const double icept = (sy - slope * sx) / m;
  double ss = 0.0;
  for (std::size_t q = 0; q < yy.size(); ++q) {
    const double r = xc[q] - (icept + slope * yy[q]);
    ss += r * r;
  }
  out.fit_rms = std::sqrt(ss / m) / dx;
  out.rho = (slope < 0.0) ? (-0.5 / slope) : std::numeric_limits<double>::infinity();
  out.valid = true;
  return out;
}

/// Two downstream tips and the grain-boundary groove of a bicrystal.
struct BicrystalTips {
  DendriteTip tip1{};
  DendriteTip tip2{};
  /// Least-downstream `+` to `-` crossing between the two seed rows: the
  /// liquid channel that lags the two grains. NaN if the seeds share a row
  /// or the channel has closed.
  double x_groove{std::numeric_limits<double>::quiet_NaN()};
  double y_groove{std::numeric_limits<double>::quiet_NaN()};
  bool valid{false};
};

/**
 * @brief Two downstream tips and the geometric GB groove between them.
 *
 * Each tip is @ref measure_downstream_tip on the y-half that contains that
 * seed, split at the midpoint so the two searches cannot steal each other's
 * arm. The groove is the *minimum* such crossing in the open interval of
 * rows between the seeds: a notch, not a Zhong grain boundary. This
 * application has one `phi`, so two solids that meet simply merge.
 */
[[nodiscard]] inline BicrystalTips
measure_bicrystal_tips(const std::vector<double> &phi_xy, int nx, int ny,
                       double dx, double dy, int i_start, int j_seed1,
                       int j_seed2, int half_width_cells) {
  BicrystalTips out;
  if (ny < 4) {
    return out;
  }
  const int ja = std::max(0, std::min(j_seed1, j_seed2));
  const int jb = std::min(ny - 1, std::max(j_seed1, j_seed2));
  const int j_mid = (ja + jb) / 2;
  const int j1_lo = (j_seed1 <= j_seed2) ? 0 : j_mid + 1;
  const int j1_hi = (j_seed1 <= j_seed2) ? j_mid : ny - 1;
  const int j2_lo = (j_seed2 < j_seed1) ? 0 : j_mid + 1;
  const int j2_hi = (j_seed2 < j_seed1) ? j_mid : ny - 1;
  out.tip1 = measure_downstream_tip(phi_xy, nx, ny, dx, dy, i_start, j1_lo,
                                    j1_hi, half_width_cells);
  out.tip2 = measure_downstream_tip(phi_xy, nx, ny, dx, dy, i_start, j2_lo,
                                    j2_hi, half_width_cells);

  auto at = [&](int i, int j) {
    return phi_xy[static_cast<std::size_t>(i) +
                  static_cast<std::size_t>(j) * static_cast<std::size_t>(nx)];
  };
  double x_min = std::numeric_limits<double>::infinity();
  int j_g = -1;
  const int i0 = std::max(0, i_start);
  for (int j = ja + 1; j <= jb - 1; ++j) {
    double xc = std::numeric_limits<double>::quiet_NaN();
    for (int i = i0; i + 1 < nx; ++i) {
      const double a = at(i, j);
      const double b = at(i + 1, j);
      if (a >= 0.0 && b < 0.0) {
        xc = (static_cast<double>(i) + a / (a - b)) * dx;
      }
    }
    if (std::isfinite(xc) && xc < x_min) {
      x_min = xc;
      j_g = j;
    }
  }
  if (j_g >= 0) {
    out.x_groove = x_min;
    out.y_groove = static_cast<double>(j_g) * dy;
  }
  out.valid = out.tip1.valid || out.tip2.valid;
  return out;
}

/// Number of tip-radius fit windows reported side by side; see
/// @ref measure_tip_scan.
inline constexpr int kTipWindowCount = 4;

/// Default fit half-widths, in cells, of @ref measure_tip_scan.
inline constexpr int kTipWindowDefaults[kTipWindowCount] = {3, 5, 8, 12};

/// Tip radius measured at several fit half-widths in the same sample.
struct TipWindowScan {
  int halfwidth[kTipWindowCount]{};
  double rho[kTipWindowCount]{};
  double fit_rms[kTipWindowCount]{};
  /// Spread `(max - min) / min` across the windows. This is *the* honesty
  /// number for a quoted `sigma*`: `sigma*` goes as `1/rho^2`, so a 20%
  /// ambiguity in `rho` is a 44% ambiguity in `sigma*`.
  double spread{std::numeric_limits<double>::quiet_NaN()};
};

/**
 * @brief Measure the tip radius at @ref kTipWindowCount fit half-widths at once.
 *
 * @details
 * The parabola half-width is a free parameter of the *measurement*, not of
 * the physics, and the previous revision of this application measured a
 * +63% spread across windows on a tip that was six cells wide. Reporting one
 * radius from one window hides that; reporting four and their spread makes
 * the ambiguity part of the result. It costs four contour scans per sample,
 * which is nothing next to a time step, and it means the sensitivity does
 * not have to be re-derived from four separate runs that might not be at the
 * same point of the transient.
 *
 * The expectation, if the tip really is a parabola resolved by the grid, is
 * that `rho` stops depending on the window: too narrow is dominated by the
 * level-set staircase and too wide reaches the non-parabolic flanks, so a
 * *flat* scan is evidence that both ends of that trade-off are far away.
 * `fit_rms` per window says which of the two is biting.
 */
[[nodiscard]] inline TipWindowScan
measure_tip_scan(const std::vector<double> &phi_xy, int nx, int ny, double dx,
                 double dy, int i_seed, int j_seed,
                 const int halfwidths[kTipWindowCount] = kTipWindowDefaults) {
  TipWindowScan out;
  double lo = std::numeric_limits<double>::infinity();
  double hi = 0.0;
  for (int q = 0; q < kTipWindowCount; ++q) {
    out.halfwidth[q] = halfwidths[q];
    const DendriteTip t =
        measure_tip(phi_xy, nx, ny, dx, dy, i_seed, j_seed, halfwidths[q]);

    out.rho[q] = t.valid ? t.rho : std::numeric_limits<double>::quiet_NaN();
    out.fit_rms[q] = t.valid ? t.fit_rms : std::numeric_limits<double>::quiet_NaN();
    if (std::isfinite(out.rho[q]) && out.rho[q] > 0.0) {
      lo = std::fmin(lo, out.rho[q]);
      hi = std::fmax(hi, out.rho[q]);
    }
  }
  if (std::isfinite(lo) && lo > 0.0) {
    out.spread = (hi - lo) / lo;
  }
  return out;
}

/**
 * @brief Fit half-widths as multiples of the tip radius itself.
 *
 * The literature convention. `measure_tip_scan` takes its windows in *cells*,
 * which is the right unit for asking whether the grid resolves the fit and
 * the wrong one for asking whether the shape is a parabola: those two
 * questions have different answers and a cell-based scan conflates them.
 *
 * Concretely: at `eps4 = 0.04` and `Omega = 0.55` the tip radius is about
 * `3 W0`. At `dx = 0.4` the widest default window, 12 cells, reaches
 * `4.8 W0` up the flank -- more than one radius past the point where a
 * dendrite stops being a paraboloid and starts being a stem. The scan then
 * reports a 49% spread that does not shrink when the grid is refined,
 * because it is not a resolution error at all; it is the flank. Measured at
 * `dx = 0.5` and `0.4` the spread was 48.9% and 49.0%, while the *velocity*
 * converged to 0.7%. A number that does not move under refinement is not
 * telling you about the discretisation.
 *
 * Scaling the windows with `rho` asks the question that has a
 * grid-independent answer, and it is the same question every published
 * tip-radius measurement asks. Two passes: window in cells to get a first
 * `rho`, then `frac * rho` converted to cells. One refinement is enough --
 * the map from window to `rho` is weak enough near the tip that a second
 * pass moves the answer by less than the spread being measured -- but a
 * pathological first pass is rejected rather than iterated on.
 *
 * @param frac  Half-widths in units of `rho`; 0.5, 1, 1.5, 2 by default.
 *              A half-width below two cells cannot support a parabola fit
 *              and reports NaN for that window rather than a number.
 */
[[nodiscard]] inline TipWindowScan
measure_tip_scan_relative(const std::vector<double> &phi_xy, int nx, int ny,
                          double dx, double dy, int i_seed, int j_seed,
                          const double frac[kTipWindowCount],
                          int seed_halfwidth = 5) {
  TipWindowScan out;
  const DendriteTip first =
      measure_tip(phi_xy, nx, ny, dx, dy, i_seed, j_seed, seed_halfwidth);
  if (!first.valid || !std::isfinite(first.rho) || first.rho <= 0.0) {
    for (int q = 0; q < kTipWindowCount; ++q) {
      out.rho[q] = std::numeric_limits<double>::quiet_NaN();
      out.fit_rms[q] = std::numeric_limits<double>::quiet_NaN();
    }
    return out;
  }
  double lo = std::numeric_limits<double>::infinity();
  double hi = 0.0;
  for (int q = 0; q < kTipWindowCount; ++q) {
    const int hw = static_cast<int>(std::lround(frac[q] * first.rho / dy));
    out.halfwidth[q] = hw;
    if (hw < 2) {
      out.rho[q] = std::numeric_limits<double>::quiet_NaN();
      out.fit_rms[q] = std::numeric_limits<double>::quiet_NaN();
      continue;
    }
    const DendriteTip t = measure_tip(phi_xy, nx, ny, dx, dy, i_seed, j_seed, hw);
    out.rho[q] = t.valid ? t.rho : std::numeric_limits<double>::quiet_NaN();
    out.fit_rms[q] = t.valid ? t.fit_rms : std::numeric_limits<double>::quiet_NaN();
    if (std::isfinite(out.rho[q]) && out.rho[q] > 0.0) {
      lo = std::fmin(lo, out.rho[q]);
      hi = std::fmax(hi, out.rho[q]);
    }
  }
  if (std::isfinite(lo) && lo > 0.0) {
    out.spread = (hi - lo) / lo;
  }
  return out;
}

/// Default relative fit half-widths, in units of `rho`.
inline constexpr double kTipWindowRelDefaults[kTipWindowCount] = {0.5, 1.0, 1.5,
                                                                  2.0};

/**
 * @brief Split-window drift of a series: is the plateau a plateau?
 *
 * Splits the trailing @p fraction of the samples in half and returns the
 * relative change of the mean between the two halves. A steady quantity
 * gives a number that shrinks as the window is pushed later; a quantity
 * still relaxing gives one that does not. Returning the *relative* change
 * rather than a slope keeps it comparable between `V` and `rho`, and makes
 * "steady to 1%" a statement rather than an impression.
 */
[[nodiscard]] inline double split_window_drift(const std::vector<double> &v,
                                               double fraction) {
  const std::size_t n = v.size();
  if (n < 8) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const std::size_t want = std::max<std::size_t>(
      8, static_cast<std::size_t>(fraction * static_cast<double>(n)));
  const std::size_t begin = (want >= n) ? 0 : (n - want);
  const std::size_t mid = begin + (n - begin) / 2;
  double a = 0.0, b = 0.0;
  double na = 0.0, nb = 0.0;
  for (std::size_t q = begin; q < mid; ++q) {
    if (std::isfinite(v[q])) {
      a += v[q];
      na += 1.0;
    }
  }
  for (std::size_t q = mid; q < n; ++q) {
    if (std::isfinite(v[q])) {
      b += v[q];
      nb += 1.0;
    }
  }
  if (!(na > 0.0) || !(nb > 0.0)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  a /= na;
  b /= nb;
  const double m = 0.5 * (a + b);
  return (m != 0.0) ? (b - a) / std::fabs(m) : std::numeric_limits<double>::quiet_NaN();
}

/**
 * @brief Same as @ref split_window_drift but for a velocity read off a
 *        position series: two independent least-squares slopes.
 *
 * Differencing the already-fitted trailing velocity would inherit the fit's
 * own window and say nothing new. Fitting `x(t)` separately over the first
 * and second half of the trailing window gives two independent estimates of
 * `V`, and their relative difference is what "the tip velocity has stopped
 * changing" has to mean operationally.
 */
[[nodiscard]] inline double velocity_split_drift(const std::vector<double> &t,
                                                 const std::vector<double> &x,
                                                 double fraction) {
  const std::size_t n = std::min(t.size(), x.size());
  if (n < 16) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const std::size_t want = std::max<std::size_t>(
      16, static_cast<std::size_t>(fraction * static_cast<double>(n)));
  const std::size_t begin = (want >= n) ? 0 : (n - want);
  const std::size_t mid = begin + (n - begin) / 2;
  const std::vector<double> t1(t.begin() + static_cast<long>(begin),
                               t.begin() + static_cast<long>(mid));
  const std::vector<double> x1(x.begin() + static_cast<long>(begin),
                               x.begin() + static_cast<long>(mid));
  const std::vector<double> t2(t.begin() + static_cast<long>(mid), t.end());
  const std::vector<double> x2(x.begin() + static_cast<long>(mid), x.end());
  const double v1 = trailing_slope(t1, x1, 1.0);
  const double v2 = trailing_slope(t2, x2, 1.0);
  if (!std::isfinite(v1) || !std::isfinite(v2)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double m = 0.5 * (v1 + v2);
  return (m != 0.0) ? (v2 - v1) / std::fabs(m)
                    : std::numeric_limits<double>::quiet_NaN();
}

/**
 * @brief Selection parameter `sigma* = 2 d0 D / (V rho^2)`.
 *
 * The quantity microscopic solvability predicts for a given anisotropy
 * strength, and the reason a tip velocity alone is not a result: `V` and
 * `rho` separately depend on the undercooling through the Ivantsov relation,
 * and only the combination is a statement about *selection*. Note the
 * `rho^2`: the relative uncertainty of `sigma*` is twice that of `rho` plus
 * that of `V`, which is why @ref TipWindowScan::spread is reported next to
 * it.
 */
[[nodiscard]] inline double selection_sigma_star(double d0, double D, double v,
                                                 double rho) noexcept {
  return (v > 0.0 && rho > 0.0) ? (2.0 * d0 * D / (v * rho * rho))
                                : std::numeric_limits<double>::quiet_NaN();
}

/**
 * @brief 2-D Ivantsov relation `Omega = sqrt(pi P) exp(P) erfc(sqrt(P))`,
 *        inverted for the tip Peclet number `P = V rho / (2 D)`.
 *
 * Reported alongside the measured `V rho` so that the two halves of the
 * selection problem can be checked separately: Ivantsov fixes the *product*
 * `V rho` from the far-field supersaturation and says nothing about how it
 * splits, and anisotropy-driven selection fixes the split. A run that
 * reproduces `sigma*` while missing `V rho` is not reproducing the physics,
 * it is cancelling two errors.
 *
 * Bisection on a monotone function; returns NaN outside `Omega` in `(0, 1)`.
 */
[[nodiscard]] inline double ivantsov_peclet_2d(double omega) {
  if (!(omega > 0.0) || !(omega < 1.0)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  auto f = [](double pe) {
    const double s = std::sqrt(pe);
    return std::sqrt(std::acos(-1.0) * pe) * std::exp(pe) * std::erfc(s);
  };
  double lo = 1.0e-12;
  double hi = 1.0;
  while (f(hi) < omega && hi < 1.0e6) {
    hi *= 2.0;
  }
  for (int it = 0; it < 200; ++it) {
    const double mid = 0.5 * (lo + hi);
    if (f(mid) < omega) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return 0.5 * (lo + hi);
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

/// `printf` into a `std::string`; keeps the CSV row builders readable.
template <class... Args>
[[nodiscard]] inline std::string format(const char *fmt, Args... args) {
  char buf[2048];
  const int n = std::snprintf(buf, sizeof(buf), fmt, args...);
  return (n > 0) ? std::string(buf, static_cast<std::size_t>(n)) : std::string();
}

} // namespace alloy_dendrite
