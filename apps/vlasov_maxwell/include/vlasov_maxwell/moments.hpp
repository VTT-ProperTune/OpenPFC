// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file moments.hpp
 * @brief Velocity-space reductions of `f_s(x, v_x, v_y)`: charge and current
 *        deposition, the kinetic conservation diagnostics, and the
 *        velocity-boundary occupancy.
 *
 * @details
 * ## What this header is for
 *
 * Everything the field solver needs from the kinetic solver, and everything
 * the CSV needs from the distribution, is an integral of `f_s` over the two
 * velocity axes:
 *
 *     rho(x)   = sum_s sigma_s int int f_s dv_x dv_y  +  rho_background
 *     J_x(x)   = sum_s sigma_s int int v_x f_s dv_x dv_y
 *     J_y(x)   = sum_s sigma_s int int v_y f_s dv_x dv_y
 *     n_s(x)   =               int int f_s dv_x dv_y
 *     W_s      = (mu_s/2) int int int |v|^2 f_s dx dv
 *     P_s      =  mu_s    int int int  v    f_s dx dv
 *
 * They share one pass over the phase-space slab, so they are computed in one
 * pass. That is not micro-optimisation: `f_s` is the largest object in the
 * application (`N_x N_vx N_vy` doubles per species, 2.1 GB at
 * `1024 x 512 x 512`), it does not fit in cache at any useful resolution, and
 * a second traversal costs a second trip through main memory for every byte
 * of it. One pass, one set of MPI reductions.
 *
 * ## Why midpoint quadrature is the right rule, and when it is not
 *
 * The velocity grid is uniform and cell-centred (@ref vlasov::SimParams::vx_of),
 * so `int g dv` is approximated by `dv sum_j g(v_j)`. For a smooth integrand
 * on a *uniform* grid the Euler-Maclaurin expansion of this rule contains only
 * boundary terms: every interior contribution cancels to all orders. If `f`
 * and all of its derivatives are at round-off on the velocity boundary -- which
 * is exactly the regime a Vlasov run must stay in to be valid at all -- those
 * boundary terms vanish too, and the rule is **spectrally accurate**, not
 * second order. The error then falls like `exp(-v_max^2 / 2 v_th^2)`
 * (truncation of the tail) and `exp(-2 pi^2 v_th^2 / dv^2)` (aliasing of the
 * Gaussian's own transform), both of which reach round-off at very modest
 * resolution. `test_fields.cpp` measures this against the analytic truncation
 * prediction `2 erfc(v_max/(sqrt2 v_th))`. At a fixed `dv = 0.1 v_th` the
 * measured relative deposition error of an analytic Maxwellian is
 *
 *     v_max/v_th :   3        4        5        6        7        8
 *     measured   : 5.4e-3   1.3e-4   1.1e-6   3.9e-9   5.0e-12  1.1e-14
 *     predicted  : 5.4e-3   1.3e-4   1.1e-6   3.9e-9   5.1e-12  2.5e-15
 *
 * -- within 2% of the prediction wherever truncation is above round-off, and
 * at round-off by `v_max = 8 v_th`.
 *
 * The converse is the honest warning. The moment the distribution has support
 * at the velocity boundary, the rule degrades to *first* order in `dv`, and the
 * deposition -- hence `rho`, hence Gauss -- degrades with it. That is why
 * @ref VelocityMoments carries @ref VelocityMoments::f_face_max and
 * @ref VelocityMoments::boundary_fraction and why the issue requires them in
 * every diagnostic sample: they are the measurement that says whether the
 * quadrature statement above still applies to the run in progress.
 *
 * ## Parallel layout
 *
 * The phase space is decomposed on `v_y` only. `x` and `v_x` are rank-local,
 * so a rank holds a full `x` line and a full `v_x` line for its slab of `v_y`
 * and every reduction here is a partial sum that must be completed across the
 * `v_y` ranks. The result is a 1-D array in `x` -- `N_x` doubles, kilobytes --
 * so it is replicated on every rank rather than distributed, and the
 * completion is an `MPI_Allreduce`.
 *
 * There are exactly three of them per call: one `MPI_SUM` over a packed buffer
 * holding the four `x` profiles and the five phase-space scalars, one `MPI_MIN`
 * for `min f`, one `MPI_MAX` for `max f` and the boundary face maximum. Packing
 * matters here because the call is per step and the payload is small enough
 * that latency, not bandwidth, is the cost.
 *
 * ## What is *not* here
 *
 * The deposition does not attempt a charge-conserving current scheme. `rho`
 * and `J` are independent quadratures of the same `f`, and discrete continuity
 * -- hence the exactness of Gauss -- holds only to the accuracy with which the
 * advection step conserves the quadrature. The issue's position is that this
 * error should be *measured*, by the Gauss residual in `maxwell.hpp`, rather
 * than designed around; this header exists to make the measurement possible,
 * not to make it come out zero.
 *
 * @see parameters.hpp for the normalisation and the grid geometry
 * @see maxwell.hpp for what consumes `rho`, `J_x` and `J_y`
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include <vlasov_maxwell/parameters.hpp>

namespace vlasov {

/**
 * @brief What a distribution has to offer to be reducible.
 *
 * Deliberately tiny. A view must answer `f(i, j, k)` for **global** indices
 * `(x, v_x, v_y)` and must be able to visit the cells this rank owns. The
 * extents come from @ref SimParams, which the caller passes anyway, so the
 * view does not have to carry them.
 *
 * Keeping the requirement this small is what lets the tests build their own
 * input -- an analytic Maxwellian in a `std::vector` -- without the phase-space
 * machinery, and what lets a GPU-resident field satisfy it later with a
 * different `for_each_owned`.
 */
template <class V>
concept DistributionView = requires(const V &f) {
  { f(0, 0, 0) } -> std::convertible_to<double>;
  f.for_each_owned([](int, int, int) {});
};

/**
 * @brief A rank's slab of `f_s` as a pointer plus strides.
 *
 * The default layout is `x` slowest, then `v_x`, then the rank-local `v_y`,
 * which is the layout that makes the `v_y` halo exchange contiguous. Any other
 * layout is expressible by setting the strides.
 *
 * `kbegin`/`kend` are the half-open range of **global** `v_y` indices this
 * rank owns; `operator()` shifts by `kbegin` internally, so callers always
 * speak in global indices.
 */
struct StridedDistribution {
  const double *data{nullptr};
  int nvx{0};                ///< global `v_x` extent (all of it is local)
  int kbegin{0};             ///< first owned global `v_y` index
  int kend{0};               ///< one past the last owned global `v_y` index
  int nx{0};                 ///< global `x` extent (all of it is local)
  std::ptrdiff_t stride_x{0};
  std::ptrdiff_t stride_vx{0};
  std::ptrdiff_t stride_vy{1};

  /// Contiguous `(x, v_x, v_y_local)` slab, `v_y` fastest.
  [[nodiscard]] static StridedDistribution contiguous(const double *data, int nx,
                                                      int nvx, int kbegin,
                                                      int kend) {
    const std::ptrdiff_t nk = static_cast<std::ptrdiff_t>(kend - kbegin);
    StridedDistribution v;
    v.data = data;
    v.nx = nx;
    v.nvx = nvx;
    v.kbegin = kbegin;
    v.kend = kend;
    v.stride_vy = 1;
    v.stride_vx = nk;
    v.stride_x = nk * static_cast<std::ptrdiff_t>(nvx);
    return v;
  }

  [[nodiscard]] double operator()(int i, int j, int k) const noexcept {
    return data[static_cast<std::ptrdiff_t>(i) * stride_x +
                static_cast<std::ptrdiff_t>(j) * stride_vx +
                static_cast<std::ptrdiff_t>(k - kbegin) * stride_vy];
  }

  template <class Fn> void for_each_owned(Fn &&fn) const {
    for (int i = 0; i < nx; ++i) {
      for (int j = 0; j < nvx; ++j) {
        for (int k = kbegin; k < kend; ++k) fn(i, j, k);
      }
    }
  }
};

/// Control for @ref reduce_velocity.
struct ReductionOptions {
  /**
   * @brief Thermal width used by the velocity-boundary occupancy.
   *
   * The occupancy counts the particle number in the shell
   * `|v_x| > v_max - v_thermal` or `|v_y| > v_max - v_thermal`. Zero (the
   * default) makes the shell empty, so the fraction is reported as zero rather
   * than as a meaningless number; a run that wants the diagnostic must say
   * what its thermal width is.
   */
  double v_thermal{0.0};
  /// Communicator spanning the `v_y` decomposition. Reductions are skipped
  /// entirely if MPI is not initialised, so a serial unit test needs no setup.
  MPI_Comm comm{MPI_COMM_WORLD};
};

/**
 * @brief All velocity-space reductions of one species, in one pass.
 *
 * The four `x` profiles are the raw moments *without* the species charge:
 * `add_species` applies `sigma_s`. Keeping them unscaled means the same object
 * serves the field sources and the per-species conservation table.
 */
struct VelocityMoments {
  int nx{0};
  std::vector<double> n;      ///< `int int f dv_x dv_y`, the number density
  std::vector<double> flux_x; ///< `int int v_x f dv_x dv_y`
  std::vector<double> flux_y; ///< `int int v_y f dv_x dv_y`
  std::vector<double> v2;     ///< `int int |v|^2 f dv_x dv_y`

  // ---- phase-space scalars (free in the same pass) ----------------------
  double number{0.0};  ///< `int int int f dx dv`, the particle number
  double l1{0.0};      ///< `int |f|`, a Casimir of the Vlasov flow
  double l2{0.0};      ///< `sqrt(int f^2)`, likewise
  double entropy{0.0}; ///< `-int f ln f` over the cells where `f > 0`
  /// Most negative value of `f`. High-order interpolation makes this nonzero;
  /// the issue requires it reported rather than clipped away.
  double f_min{0.0};
  double f_max{0.0};

  // ---- velocity-boundary occupancy --------------------------------------
  /**
   * @brief `max |f|` over the four velocity boundary faces
   *        (`j = 0`, `j = nvx-1`, `k = 0`, `k = nvy-1`).
   *
   * The zero-inflow velocity boundary is a lie that is only as good as this
   * number is small. Compare it against @ref f_max.
   */
  double f_face_max{0.0};
  /**
   * @brief Share of the particle number sitting within one thermal width of a
   *        velocity boundary.
   *
   * Dimensionless, so the phase-space volume element cancels and it is
   * meaningful at any resolution. A run in which this grows is not a valid run.
   */
  double boundary_fraction{0.0};

  /// `f_face_max` relative to the peak of `f`. The scale-free form of the
  /// same statement; `1e-12` is comfortable, `1e-4` is a run to stop.
  [[nodiscard]] double face_ratio() const noexcept {
    return f_face_max / std::max(std::fabs(f_max), kTiny);
  }
};

/**
 * @brief Reduce one species over `(v_x, v_y)` and complete the sum across the
 *        `v_y` ranks.
 *
 * Midpoint quadrature on the uniform velocity grid; see the file comment for
 * why that is spectrally accurate here and for the circumstances in which it
 * stops being.
 *
 * @param p    grid geometry (extents, `v_max`, spacings)
 * @param f    this rank's slab of `f_s`
 * @param opt  thermal width for the occupancy diagnostic, and the communicator
 */
template <DistributionView V>
[[nodiscard]] inline VelocityMoments
reduce_velocity(const SimParams &p, const V &f, const ReductionOptions &opt = {}) {
  const int nx = p.nx;
  const int nvx = p.nvx;
  const int nvy = p.nvy;
  if (nx <= 0 || nvx <= 0 || nvy <= 0) {
    throw std::invalid_argument("reduce_velocity: empty grid");
  }

  std::vector<double> vxs(static_cast<std::size_t>(nvx));
  std::vector<double> vys(static_cast<std::size_t>(nvy));
  for (int j = 0; j < nvx; ++j) vxs[static_cast<std::size_t>(j)] = p.vx_of(j);
  for (int k = 0; k < nvy; ++k) vys[static_cast<std::size_t>(k)] = p.vy_of(k);

  // The shell is empty when v_thermal is zero, because no cell centre reaches
  // v_max: the outermost centre is at v_max - dv/2.
  const double edge = p.v_max - opt.v_thermal;

  // Packed reduction buffer: [n | flux_x | flux_y | v2 | total shell l1 l2 S].
  constexpr int kScalars = 5;
  const std::size_t nxs = static_cast<std::size_t>(nx);
  std::vector<double> acc(4 * nxs + kScalars, 0.0);
  double *an = acc.data();
  double *ax = an + nxs;
  double *ay = ax + nxs;
  double *ae = ay + nxs;
  double *sc = ae + nxs; // total, shell, l1, l2sq, entropy

  double fmin = std::numeric_limits<double>::infinity();
  double fmax = -std::numeric_limits<double>::infinity();
  double fface = 0.0;

  f.for_each_owned([&](int i, int j, int k) {
    const double val = f(i, j, k);
    const double vx = vxs[static_cast<std::size_t>(j)];
    const double vy = vys[static_cast<std::size_t>(k)];
    const std::size_t ii = static_cast<std::size_t>(i);
    an[ii] += val;
    ax[ii] += vx * val;
    ay[ii] += vy * val;
    ae[ii] += (vx * vx + vy * vy) * val;
    sc[0] += val;
    if (std::fabs(vx) > edge || std::fabs(vy) > edge) sc[1] += val;
    sc[2] += std::fabs(val);
    sc[3] += val * val;
    if (val > 0.0) sc[4] -= val * std::log(val);
    fmin = std::min(fmin, val);
    fmax = std::max(fmax, val);
    if (j == 0 || j == nvx - 1 || k == 0 || k == nvy - 1) {
      fface = std::max(fface, std::fabs(val));
    }
  });

  int inited = 0;
  MPI_Initialized(&inited);
  if (inited != 0 && opt.comm != MPI_COMM_NULL) {
    MPI_Allreduce(MPI_IN_PLACE, acc.data(), static_cast<int>(acc.size()),
                  MPI_DOUBLE, MPI_SUM, opt.comm);
    MPI_Allreduce(MPI_IN_PLACE, &fmin, 1, MPI_DOUBLE, MPI_MIN, opt.comm);
    std::array<double, 2> hi{fmax, fface};
    MPI_Allreduce(MPI_IN_PLACE, hi.data(), 2, MPI_DOUBLE, MPI_MAX, opt.comm);
    fmax = hi[0];
    fface = hi[1];
  }

  const double dv = p.dvx() * p.dvy();
  const double dV = p.dx() * dv;

  VelocityMoments m;
  m.nx = nx;
  m.n.resize(nxs);
  m.flux_x.resize(nxs);
  m.flux_y.resize(nxs);
  m.v2.resize(nxs);
  for (std::size_t i = 0; i < nxs; ++i) {
    m.n[i] = an[i] * dv;
    m.flux_x[i] = ax[i] * dv;
    m.flux_y[i] = ay[i] * dv;
    m.v2[i] = ae[i] * dv;
  }
  m.number = sc[0] * dV;
  m.l1 = sc[2] * dV;
  m.l2 = std::sqrt(std::max(sc[3], 0.0) * dV);
  m.entropy = sc[4] * dV;
  m.f_min = std::isfinite(fmin) ? fmin : 0.0;
  m.f_max = std::isfinite(fmax) ? fmax : 0.0;
  m.f_face_max = fface;
  m.boundary_fraction = sc[1] / (std::fabs(sc[0]) > kTiny ? sc[0] : kTiny);
  return m;
}

/// `int int int f dx dv` for one species. Identical to
/// @ref VelocityMoments::number; provided for symmetry with the others.
[[nodiscard]] inline double number(const VelocityMoments &m) noexcept {
  return m.number;
}

/// `(mu_s/2) int int int |v|^2 f_s dx dv`, the species kinetic energy.
[[nodiscard]] inline double kinetic_energy(const SimParams &p, const Species &s,
                                           const VelocityMoments &m) noexcept {
  double acc = 0.0;
  for (double e : m.v2) acc += e;
  return 0.5 * s.mu * acc * p.dx();
}

/// `mu_s int int int v f_s dx dv`, the species momentum, as `(P_x, P_y)`.
[[nodiscard]] inline std::array<double, 2>
momentum(const SimParams &p, const Species &s, const VelocityMoments &m) noexcept {
  double px = 0.0;
  double py = 0.0;
  for (std::size_t i = 0; i < m.flux_x.size(); ++i) {
    px += m.flux_x[i];
    py += m.flux_y[i];
  }
  const double w = s.mu * p.dx();
  return {px * w, py * w};
}

/**
 * @brief The 1-D field sources, replicated on every rank.
 *
 * `N_x` doubles each: too small to be worth distributing, and replicating them
 * is what lets the field solve be a rank-local serial computation with no
 * transpose in it.
 */
struct Sources {
  std::vector<double> rho;
  std::vector<double> Jx;
  std::vector<double> Jy;
  /// Background actually applied by @ref apply_background.
  double background{0.0};
  /// Mean of `rho` after the background. The `k = 0` mode of Gauss is solvable
  /// only if this is zero; see @ref vlasov::solve_gauss.
  double net_charge{0.0};

  [[nodiscard]] static Sources zeros(int nx) {
    Sources s;
    s.rho.assign(static_cast<std::size_t>(nx), 0.0);
    s.Jx.assign(static_cast<std::size_t>(nx), 0.0);
    s.Jy.assign(static_cast<std::size_t>(nx), 0.0);
    return s;
  }
};

/**
 * @brief Accumulate one species into the field sources.
 *
 * `rho += sigma_s n_s`, `J += sigma_s (flux_x, flux_y)`. Charge enters here and
 * only here, so a two-species run is `add_species` twice with the same code.
 */
inline void add_species(const Species &s, const VelocityMoments &m, Sources &out) {
  if (out.rho.size() != m.n.size()) {
    throw std::invalid_argument("add_species: size mismatch between sources and moments");
  }
  for (std::size_t i = 0; i < m.n.size(); ++i) {
    out.rho[i] += s.sigma * m.n[i];
    out.Jx[i] += s.sigma * m.flux_x[i];
    out.Jy[i] += s.sigma * m.flux_y[i];
  }
}

/**
 * @brief Apply the neutralising background and report the net charge.
 *
 * `SimParams::rho_background` is NaN by default, which means "choose the value
 * that makes the state exactly neutral" -- the fixed neutralising ion
 * background every electrostatic and Weibel benchmark assumes. An explicit
 * value is honoured *as given*: the resulting net charge is then reported in
 * @ref Sources::net_charge rather than corrected, because a run that is not
 * neutral has no periodic solution of Gauss and the reader is entitled to find
 * that out from the diagnostics instead of from a wrong answer.
 *
 * @return the background density actually applied
 */
inline double apply_background(const SimParams &p, Sources &out) {
  const std::size_t n = out.rho.size();
  if (n == 0) throw std::invalid_argument("apply_background: empty sources");
  double mean = 0.0;
  for (double r : out.rho) mean += r;
  mean /= static_cast<double>(n);

  const double bg = std::isnan(p.rho_background) ? -mean : p.rho_background;
  for (double &r : out.rho) r += bg;

  out.background = bg;
  out.net_charge = mean + bg;
  return bg;
}

/**
 * @brief One-species convenience: reduce, deposit, neutralise.
 *
 * The multi-species path is @ref reduce_velocity + @ref add_species per
 * species followed by a single @ref apply_background; this wrapper exists
 * because the default configuration is one electron species on a fixed
 * background and spelling that out at every call site invites a mistake in
 * the ordering (the background must be applied once, after every species).
 */
template <DistributionView V>
[[nodiscard]] inline Sources deposit(const SimParams &p, const Species &s,
                                     const V &f, VelocityMoments &m_out,
                                     const ReductionOptions &opt = {}) {
  m_out = reduce_velocity(p, f, opt);
  Sources src = Sources::zeros(p.nx);
  add_species(s, m_out, src);
  apply_background(p, src);
  return src;
}

} // namespace vlasov
