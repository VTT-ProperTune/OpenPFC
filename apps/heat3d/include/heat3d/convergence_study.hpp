// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file convergence_study.hpp
 * @brief Spatial order-of-accuracy measurement for the compact FD Laplacian.
 *
 * @details
 * `heat3d_fd` ships central FD stencils at every even order 2..20 (see
 * `pfc::gradient::FDGradient<G>`), but nothing in the repository ever
 * measured whether raising the order actually buys accuracy. This header
 * is the shared "run one (order, N) case and return its L2 error" building
 * block used by both the `heat3d_fd_convergence_study` driver (the full
 * sweep -> CSV) and `tests/test_heat3d_fd_convergence.cpp` (a regression
 * guard on two representative orders).
 *
 * ## Isolating spatial from temporal error: use *no* time discretization
 *
 * The textbook risk in any such study is described in
 * `tests/integration/scenarios/time_integration/test_rk3_convergence.cpp`
 * for the mirror-image problem (isolating *temporal* order from spatial
 * contamination): pick the wrong time step and you measure the time
 * integrator, not the stencil. Two explicit-Euler designs were tried here
 * and both failed that test for exactly the reason it warns about, before
 * landing on the one below that sidesteps the question entirely.
 *
 *  - **Attempt 1**: `dt = dt_safety * dx^2 / kD`, i.e. the standard
 *    parabolic-CFL scaling, re-derived at *each* case's own `dx`. Euler's
 *    `O(dt)` global temporal error then scales as `dx^2` -- the same
 *    power as the plain 7-point stencil's own *design* order -- so
 *    orders 4/6/8/10/12 all flattened onto one common `dx^2`-shaped floor
 *    at fine grids instead of showing their own higher-power curves: the
 *    curves-all-flatten-at-once symptom, verified by running (see the git
 *    history of this file / the PR that introduced it).
 *  - **Attempt 2**: one *fixed* `dt` for the whole sweep (independent of
 *    `dx`), sized from the finest grid's stability limit with a safety
 *    margin -- removing the `dx^2` *growth* of the floor at coarse grids.
 *    Measured floor: ~1.8e-6, matching the predicted
 *    `~0.5 * D^2 * m^4 * t * dt * exp(-D m^2 t)` for that `dt`. Reaching a
 *    floor near round-off (`~1e-13`) this way needs `dt` smaller by
 *    ~5 orders of magnitude at *fixed* `kFinalTime` -- i.e. ~10^5x more
 *    steps, since `n_steps = kFinalTime / dt` is one number shared by
 *    every case in the sweep. That is minutes turning into the better
 *    part of a day for no benefit: a fixed, non-vanishing `dt` was always
 *    going to leave *some* floor, just a lower one.
 *  - **What is used now**: no time-stepping loop at all. `HeatModel`'s
 *    single-mode IC, `\f$u_0(x,y,z) = \cos(mx)\f$` with integer mode
 *    number `m` on a periodic domain of length `kDomainLength = 2*pi`, is
 *    an *exact eigenfunction* of `pfc::gradient::FDGradient<HeatGrads>`'s
 *    stencil under periodic wraparound: any linear, shift-invariant
 *    (i.e. constant-coefficient, periodic) stencil is diagonalized by the
 *    discrete Fourier basis, a standard circulant-matrix fact, so
 *    `Lap_FD[cos(m x_i)] = -k_eff^2 cos(m x_i)` for the *same* `k_eff` at
 *    every grid point `i` (up to round-off) -- `run_case()` verifies this
 *    numerically rather than assuming it (see `eigenvalue_residual`
 *    below). `k_eff` differs from the true `m` by exactly the stencil's
 *    own dispersion error, which is what this study is trying to measure.
 *    Because `u` stays proportional to `cos(m x)` for **all** time under
 *    the FD-only (not yet time-discretized) semi-discrete ODE, that ODE
 *    has a closed-form solution, `u(x, t) = u_0(x) * exp(kD * (-k_eff^2)
 *    * t)`, evaluated directly with **zero** time-discretization error --
 *    not "small", exactly zero, by construction, for any `t` -- rather
 *    than approximated by marching a stepper forward. This is the same
 *    idea `test_rk3_convergence.cpp` uses in the opposite direction
 *    (replace the FD *spatial* operator by its *exact* eigenvalue action
 *    to isolate a *temporal* order; here the *time* evolution is replaced
 *    by *its* exact action to isolate a *spatial* order). What remains
 *    after squaring `u(x, kFinalTime)` against the true PDE solution
 *    `exact_single_mode` is purely the stencil's spatial truncation error.
 *
 * `kD` here is `heat3d::kD`, matching every other heat3d binary.
 */

#include <cmath>
#include <cstddef>
#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

#include <heat3d/heat_model.hpp>

namespace heat3d::convergence {

/// Domain length (periodic box, all three axes): one integer mode number
/// spans this exactly, at every grid resolution.
inline constexpr double kDomainLength = 2.0 * M_PI;

/// Integer spatial mode number for the single-Fourier-mode IC `cos(m*x)`.
inline constexpr double kMode = 3.0;

/// Physical final time. Chosen so the mode decays by a non-trivial,
/// non-extreme factor: `exp(-kD * kMode^2 * kFinalTime) ~ 0.64`.
inline constexpr double kFinalTime = 0.05;

/// Exact PDE solution for the single-mode IC: `u(x,t) = exp(-D m^2 t) cos(m x)`.
[[nodiscard]] inline double exact_single_mode(double x, double t) noexcept {
  return std::exp(-kD * kMode * kMode * t) * std::cos(kMode * x);
}

/// One (fd_order, N) measurement.
struct ConvergenceCase {
  int fd_order{2};
  int N{16};
  double dx{0.0};
  /// Numerical eigenvalue `-k_eff^2` the stencil produced for `cos(m*x)`;
  /// the true continuous value is `-kMode^2`.
  double eigenvalue{0.0};
  /// RMS of `Lap_FD[u0](x) - eigenvalue * u0(x)` over `u0`'s own RMS: how
  /// far `cos(m*x)` is from being an *exact* eigenvector of this stencil
  /// on this grid, in floating point. Expected to be `~1e-15` (round-off)
  /// -- see the file header. A value that is not tiny would mean the
  /// "closed-form time evolution" step below is not actually exact for
  /// this case, and should be treated as a red flag, not as data.
  double eigenvalue_residual{0.0};
  double l2_error{0.0};
};

/**
 * @brief Measure the L2 (RMS) error of the FD Laplacian's single-mode
 *        eigenvalue, evolved *exactly* in time to `kFinalTime`, against
 *        the true PDE solution -- see the file header for why this has
 *        zero time-discretization error by construction.
 *
 * Single-rank by construction (matches the pattern used throughout
 * `apps/heat3d/tests/test_heat3d.cpp`): a convergence study needs
 * reproducible numbers, not parallel scaling.
 *
 * @param fd_order  Even FD order 2..20 (halo width = fd_order/2).
 * @param N         Grid points per axis (must be > fd_order/2).
 */
[[nodiscard]] inline ConvergenceCase run_case(int fd_order, int N) {
  ConvergenceCase result;
  result.fd_order = fd_order;
  result.N = N;

  const double dx = kDomainLength / static_cast<double>(N);
  result.dx = dx;

  // Geometry + decomposition, single rank -- same calls as heat3d_fd.cpp
  // step 2, just with dx = kDomainLength/N instead of a fixed unit spacing.
  const auto domain =
      pfc::domain::create(pfc::GridSize({N, N, N}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                          pfc::GridSpacing({dx, dx, dx}));
  const auto decomp = pfc::decomposition::create(domain, /*nproc=*/1);

  // Storage + halo exchanger + gradient evaluator: heat3d_fd.cpp steps 3-4.
  // `u` is *padded* (storage_halo == hw), so FDGradient's padded-Field
  // constructor binds imin/imax to the whole owned domain [0, N) -- see
  // apps/heat3d/README.md for why this matters (an earlier version of
  // this study bound FDGradient to FDCPUStack's *unpadded* field instead,
  // which silently drops a fd_order/2-wide boundary shell; invisible for
  // a Gaussian IC that decays to ~0 at the domain edge, not for this
  // full-amplitude periodic one).
  const int hw = fd_order / 2;
  pfc::data::Field<double, pfc::HostSpace> u =
      pfc::data::field_from_subdomain<double>(decomp, /*rank=*/0, hw);
  pfc::data::Field<double, pfc::HostSpace> lap =
      pfc::data::field_from_subdomain<double>(decomp, /*rank=*/0, hw);
  pfc::comm::HaloExchange<pfc::HostSpace, double> halo(u, decomp, /*rank=*/0,
                                                       MPI_COMM_WORLD);
  pfc::gradient::FDGradient<HeatGrads> grad(u, fd_order);

  HeatModel model;
  model.initial_condition = [](double x, double, double) {
    return std::cos(kMode * x);
  };
  u.apply(model.initial_condition); // u = u0, never mutated below.
  halo.exchange();                 // fill the periodic ghost ring once.

  lap.for_each_owned([&](int i, int j, int k) {
    const auto g = pfc::gradient::evaluate(grad, pfc::Int3{i, j, k});
    lap(i, j, k) = g.xx + g.yy + g.zz;
  });

  // Least-squares eigenvalue: Lap_FD[u0] should equal eigenvalue * u0
  // pointwise (see file header). Projecting rather than dividing
  // pointwise sidesteps u0's zero crossings and is exact in the presence
  // of round-off noise on either side.
  double num = 0.0, den = 0.0;
  u.for_each_owned([&](int i, int j, int k) {
    const double u0 = u(i, j, k);
    num += lap(i, j, k) * u0;
    den += u0 * u0;
  });
  const double eigenvalue = num / den;
  result.eigenvalue = eigenvalue;

  double resid_sq = 0.0;
  u.for_each_owned([&](int i, int j, int k) {
    const double d = lap(i, j, k) - eigenvalue * u(i, j, k);
    resid_sq += d * d;
  });
  result.eigenvalue_residual = std::sqrt(resid_sq / den);

  // Exact (non-discretized) time evolution of du/dt = kD * eigenvalue * u:
  // u(kFinalTime) = u0 * exp(kD * eigenvalue * kFinalTime), applied
  // directly -- no stepping loop, so no dt and no temporal truncation
  // error at all.
  const double decay = std::exp(kD * eigenvalue * kFinalTime);

  double sum_sq = 0.0;
  double count = 0.0;
  u.for_each_owned([&](double x, double, double, double u0) {
    const double u_fd = u0 * decay;
    const double diff = u_fd - exact_single_mode(x, kFinalTime);
    sum_sq += diff * diff;
    count += 1.0;
  });
  result.l2_error = std::sqrt(sum_sq / count);
  return result;
}

/**
 * @brief Observed order of accuracy between two consecutive-N cases at the
 *        same fd_order, via `p = log(e1/e2) / log(dx1/dx2)`.
 *
 * @param coarser Case with the larger `dx` (fewer grid points).
 * @param finer   Case with the smaller `dx` (more grid points).
 */
[[nodiscard]] inline double observed_order(const ConvergenceCase &coarser,
                                           const ConvergenceCase &finer) noexcept {
  return std::log(coarser.l2_error / finer.l2_error) /
         std::log(coarser.dx / finer.dx);
}

} // namespace heat3d::convergence
