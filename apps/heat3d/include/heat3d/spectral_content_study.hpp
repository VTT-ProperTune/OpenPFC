// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spectral_content_study.hpp
 * @brief Where a spectral Laplacian is cheaper than a finite-difference one,
 *        as a function of how much of Nyquist the field's content reaches.
 *
 * @details
 * ## Why this exists
 *
 * `docs/report/16_scalability.qmd` measures the spectral and
 * finite-difference paths of @sec-heat3d on two of the three axes that
 * decide between them — cost per step at equal grid, and parallel
 * efficiency at equal grid — and then declines to give a recommendation,
 * because the third axis, **accuracy**, was measured only by
 * `heat3d_fd_convergence_study`, which evolves a *single Fourier mode*.
 *
 * A single smooth mode is the most favourable case a high-order stencil
 * can be handed: the stencil's error constant multiplies a high derivative
 * that one smooth mode barely has. Read literally, those numbers said FD-4
 * reaches \f$L^2 = 10^{-6}\f$ at \f$N=129\f$ for 0.02 ms against the
 * spectral path's 214 ms — a verdict that does not transfer to a PFC
 * crystal carrying real content at \f$2k_0\f$ and \f$3k_0\f$
 * (`apps/tungsten/include/tungsten/resolution.hpp`), or to a dendrite tip.
 * This header closes that gap by measuring accuracy for a field of
 * *controlled, arbitrary* spectral content instead of one mode.
 *
 * ## Why a closed form is available at all
 *
 * `heat3d` solves \f$\partial_t u = D\nabla^2 u\f$ on a periodic box. That
 * is **linear and constant-coefficient**, so the discrete Fourier basis
 * diagonalises every operator in sight and each mode evolves independently:
 *
 *  - the true PDE decays mode \f$\mathbf k\f$ as \f$e^{-D|\mathbf k|^2 t}\f$;
 *  - the semi-discrete FD system decays it as \f$e^{D\lambda_p(\mathbf k)t}\f$,
 *    where \f$\lambda_p\f$ is the stencil's own symbol (below);
 *  - the spectral operator uses \f$-|\mathbf k|^2\f$ itself, so its spatial
 *    error is **identically zero**, not merely small.
 *
 * By Parseval the \f$L^2\f$ error of *any* initial field is therefore a
 * closed-form sum over its own spectrum \f$|a_{\mathbf k}|^2\f$ — no
 * simulation needed. `heat3d_spectral_content_study` sweeps that sum; the
 * same driver then *validates* it against real time-stepped runs of the
 * production FD stack (`run_validation()` below), because a semi-analytic
 * result nobody checked is a guess with more decimal places.
 *
 * ## The FD symbol, and how it is computed without cancellation
 *
 * `pfc::field::fd::EvenCentralD2<Order>` stores the central stencil as
 * integer weights over a common denominator:
 * \f$ \partial_x^2 u \approx (c_0 u_0 + \sum_{j\ge1} c_j(u_{-j}+u_{+j}))
 *     / (D_2 h^2) \f$.
 * Applied to \f$e^{i\kappa x}\f$ this gives the symbol
 * \f$ \lambda_p(\kappa)h^2 = (c_0 + 2\sum_j c_j\cos(j\theta))/D_2 \f$ with
 * \f$\theta = \kappa h\f$, and the exact value is \f$-\theta^2\f$. What the
 * error map needs is the **defect** \f$\theta^2 + \lambda_p h^2\f$, which
 * for a high order at small \f$\theta\f$ is fourteen orders of magnitude
 * below the two terms being subtracted — evaluated naively it is pure
 * round-off (order 12 at \f$\theta = 0.1\f$: the direct difference gives
 * `3.4e-16`, the true value is `1.2e-19`).
 *
 * The way out is an identity worth knowing. With
 * \f$\delta^2 = 2-2\cos\theta\f$ (i.e. \f$\delta = 2\sin(\theta/2)\f$, the
 * symbol of the *plain* second difference),
 * \f[
 *   \theta^2 = \bigl(2\arcsin(\delta/2)\bigr)^2
 *            = \sum_{m\ge1} a_m \delta^{2m}, \qquad
 *   a_m = \frac{2}{m^2\binom{2m}{m}},
 * \f]
 * and the order-\f$2M\f$ central stencil's symbol is **exactly** this
 * series truncated at \f$m = M\f$ (verified against the shipped tables in
 * `tests/test_heat3d_spectral_content.cpp`). The defect is therefore the
 * *tail*
 * \f$ \theta^2 + \lambda_p h^2 = \sum_{m>M} a_m\delta^{2m} \f$ —
 * a sum of strictly positive terms, so it is computed to full relative
 * precision with no cancellation at all. The tail converges like
 * \f$(\delta^2/4)^m\f$, which is fast for \f$\theta \lesssim \pi/2\f$ and
 * slow at \f$\theta\to\pi\f$; there the defect is a large fraction of
 * \f$\theta^2\f$ and the direct difference keeps ten digits, so
 * `fd_symbol_defect()` switches to it.
 *
 * ## The initial-condition family, and what "fraction of Nyquist" means
 *
 * A Gaussian of width \f$\sigma\f$ has a Gaussian spectrum of width
 * \f$1/\sigma\f$, so one parameter sweeps continuously from "all content at
 * low \f$k\f$" to "content up against Nyquist". The field used here is the
 * separable, Nyquist-truncated periodic Gaussian
 * \f$u_0 = g(x)g(y)g(z)\f$ with
 * \f$g(x) = \sum_{|n|\le N/2-1} e^{-\sigma^2 n^2/2}e^{inx}\f$ on
 * \f$L = 2\pi\f$. Building it from its own Fourier series rather than
 * sampling \f$e^{-r^2/2\sigma^2}\f$ matters: the sampled continuum Gaussian
 * aliases at the \f$10^{-6}\f$ level in energy, which would sit right on top
 * of the errors being measured.
 *
 * **Definition of the content fraction.** \f$f\f$ is the fraction of the
 * Nyquist wavenumber \f$k_\mathrm{Nyq} = \pi/h\f$ at which the initial
 * *amplitude* spectrum has fallen to `kContentThreshold` \f$=10^{-3}\f$ of
 * its peak:
 * \f[
 *   |a_k|/|a_0| = 10^{-3} \ \text{at}\ k = k_c \equiv f\,k_\mathrm{Nyq},
 *   \qquad
 *   \sigma = \sqrt{2\ln 10^{3}}\,/\,k_c .
 * \f]
 * So \f$f=0.2\f$ is a smooth blob on a grid four to five times finer than
 * it needs, and \f$f=0.9\f$ is a field whose content runs right up to the
 * grid scale. There is nothing special about \f$10^{-3}\f$ beyond being
 * stated: a different threshold rescales \f$f\f$ by a constant and moves
 * every curve together.
 *
 * **Dimensionless time.** The remaining parameter is how long the field is
 * diffused, as \f$\tau = D\,t\,k_c^2\f$ — the content-edge mode decays by
 * \f$e^{-\tau}\f$. `kDiffusionTime` \f$=1\f$ throughout. The map is
 * insensitive to it in the way that matters: over
 * \f$\tau \in [0.5, 2]\f$ every error moves by less than a factor two, and
 * the *crossover* result below does not depend on \f$\tau\f$ at all.
 *
 * With those two choices the error map is a function of
 * \f$(p, f, \tau)\f$ **only** — the grid \f$N\f$ drops out, because
 * \f$\theta = \pi f (n/k_c)\f$ and the spectrum depends on \f$n/k_c\f$
 * alone. `predict_l2_error()` still takes an `N` (it sums over a real
 * \f$N^3\f$ mode cube, which is what makes it directly comparable with a
 * run), and `tests/test_heat3d_spectral_content.cpp` pins the
 * \f$N\f$-independence rather than assuming it.
 *
 * ## The crossover, and the one modelling assumption in it
 *
 * At a target error \f$\varepsilon\f$, method \f$p\f$ may use the coarsest
 * grid on which it still meets \f$\varepsilon\f$ — equivalently the largest
 * content fraction \f$f^*_p(\varepsilon)\f$ it tolerates
 * (`content_fraction_at()`). The spectral path has no accuracy constraint
 * at all here, only a representation one, so \f$f^*_\mathrm{spec} = 1\f$.
 * Since \f$k_c\f$ is fixed by the physics, \f$N \propto 1/f^*\f$, and
 * **assuming the cost of a fixed number of steps scales as \f$N^3\f$** the
 * equal-accuracy cost is \f$c_p/(f^*_p)^3\f$ against \f$c_\mathrm{spec}\f$
 * for the spectral path. Hence
 * \f[
 *   \text{FD-}p \text{ is cheaper} \iff
 *   f^*_p(\varepsilon) > \sqrt[3]{c_p/c_\mathrm{spec}} .
 * \f]
 * That threshold (`crossover_fraction()`) is **pure cost arithmetic** —
 * independent of \f$\tau\f$, of the threshold convention, and of the
 * numerics. With the measured costs in `docs/report/data/heat3d_method_cost.csv`
 * it sits between 0.32 (FD-2) and 0.49 (FD-12): a 32x per-step advantage
 * buys only a factor \f$32^{1/3} = 3.2\f$ in grid spacing, which is much
 * less protection than the raw cost table suggests.
 *
 * The \f$N^3\f$ model counts a **fixed number of steps**, which is
 * generous to finite differences and deliberately so: `heat3d_fd` is
 * explicit, so its stable \f$\Delta t\f$ also falls as \f$\Delta x^2\f$,
 * while `heat3d_spectral` is unconditionally stable implicit Euler. At
 * fixed physical end time the FD refinement penalty is nearer \f$N^5\f$
 * than \f$N^3\f$. Every crossover reported here is therefore a *lower*
 * bound on how often the spectral path wins.
 *
 * @see convergence_study.hpp — the single-mode study this one generalises.
 * @see openpfc/kernel/field/fd_stencils.hpp — the stencil tables whose
 *      symbol is reproduced here.
 * @see apps/tungsten/include/tungsten/resolution.hpp — the *nonlinear*
 *      half of the same question (aliasing of a cubic term), measured
 *      separately; nothing here says anything about it.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/fd_stencils.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

#include <heat3d/heat_model.hpp>

namespace heat3d::spectral_content {

/// Periodic box length on every axis, so mode numbers are plain integers.
inline constexpr double kDomainLength = 2.0 * M_PI;

/// Amplitude ratio defining the "content edge" \f$k_c\f$ (see file header).
inline constexpr double kContentThreshold = 1.0e-3;

/// Dimensionless diffusion time \f$\tau = D\,t\,k_c^2\f$ used everywhere.
inline constexpr double kDiffusionTime = 1.0;

/// Grid used for the reported semi-analytic sweep. The map is
/// \f$N\f$-independent above ~48 (pinned by the unit test); 128 is chosen
/// so even the smallest \f$f\f$ in the sweep still has a well-sampled
/// spectrum.
inline constexpr int kMapGrid = 128;

/// \f$\sigma^2 k_c^2 = 2\ln(1/\text{threshold})\f$: the one constant tying
/// the Gaussian width to the content fraction.
[[nodiscard]] inline double content_shape_constant() noexcept {
  return 2.0 * std::log(1.0 / kContentThreshold);
}

/**
 * @brief Coefficient \f$a_m = 2/(m^2\binom{2m}{m})\f$ of
 *        \f$(2\arcsin(\delta/2))^2 = \sum_m a_m \delta^{2m}\f$.
 *
 * Evaluated by the recurrence \f$a_{m+1} = a_m\,m^2/\bigl((m+1)\,2(2m+1)\bigr)\f$
 * from \f$a_1 = 1\f$, because \f$\binom{2m}{m}\f$ overflows a `double`
 * long before the series stops mattering.
 */
[[nodiscard]] inline double arcsin_series_coefficient(int m) {
  if (m < 1) throw std::invalid_argument("arcsin_series_coefficient: m >= 1");
  double a = 1.0;
  for (int i = 1; i < m; ++i) {
    a *= static_cast<double>(i) * static_cast<double>(i) /
         (static_cast<double>(i + 1) * 2.0 * static_cast<double>(2 * i + 1));
  }
  return a;
}

/**
 * @brief Symbol of the tabulated central second-derivative stencil:
 *        \f$\lambda_p(\kappa)\,h^2\f$ at \f$\theta = \kappa h\f$.
 *
 * Reads the *shipped* table via `lookup_even_central_d2`, so this is the
 * symbol of the stencil `pfc::gradient::FDGradient<HeatGrads>` actually
 * applies, not a re-derivation of it.
 *
 * @param order Even FD order with a tabulated D2 stencil (2..20).
 * @param theta \f$\kappa h\f$, normally in \f$[0,\pi]\f$.
 * @throws std::invalid_argument if `order` is not tabulated.
 */
[[nodiscard]] inline double fd_symbol(int order, double theta) {
  pfc::field::fd::EvenCentralD2View view{};
  if (!pfc::field::fd::lookup_even_central_d2(order, &view)) {
    throw std::invalid_argument("fd_symbol: no tabulated D2 stencil for order " +
                                std::to_string(order));
  }
  double sum = static_cast<double>(view.coeffs[0]);
  for (int j = 1; j <= view.half_width; ++j) {
    sum += 2.0 * static_cast<double>(view.coeffs[j]) *
           std::cos(static_cast<double>(j) * theta);
  }
  return sum / static_cast<double>(view.denom);
}

/**
 * @brief The stencil's dispersion defect \f$\theta^2 + \lambda_p h^2 \ge 0\f$.
 *
 * This is the whole spatial error of the FD Laplacian for one mode: the
 * stencil always *under*-estimates \f$|\lambda|\f$, so an FD-evolved mode
 * decays too slowly by exactly this amount. Computed from the positive-term
 * arcsin tail (see the file header) wherever that converges, which is what
 * makes the map trustworthy down to \f$10^{-19}\f$ instead of dying in
 * cancellation at \f$10^{-16}\f$.
 *
 * @param order Even FD order 2..20.
 * @param theta \f$\kappa h \in [0,\pi]\f$.
 */
[[nodiscard]] inline double fd_symbol_defect(int order, double theta) {
  const double delta_sq = 2.0 - 2.0 * std::cos(theta);
  // Beyond this the tail converges too slowly to be worth it -- and there
  // the defect is a large enough fraction of theta^2 that the direct
  // difference below still keeps ~10 significant digits.
  constexpr double kSeriesLimit = 2.5;
  if (delta_sq <= kSeriesLimit) {
    const int half_width = order / 2;
    double a = arcsin_series_coefficient(half_width);
    double power = std::pow(delta_sq, static_cast<double>(half_width));
    double total = 0.0;
    for (int m = half_width; m < half_width + 600; ++m) {
      // Advance to term m+1.
      a *= static_cast<double>(m) * static_cast<double>(m) /
           (static_cast<double>(m + 1) * 2.0 * static_cast<double>(2 * m + 1));
      power *= delta_sq;
      const double term = a * power;
      total += term;
      if (term <= 1.0e-18 * total) return total;
    }
    return total;
  }
  return theta * theta + fd_symbol(order, theta);
}

/**
 * @brief Gaussian width \f$\sigma\f$ (in the \f$L=2\pi\f$ box) whose content
 *        edge sits at fraction `f` of Nyquist on an `N` grid.
 */
[[nodiscard]] inline double content_sigma(double f, int N) {
  const double k_nyquist = M_PI / (kDomainLength / static_cast<double>(N));
  return std::sqrt(content_shape_constant()) / (f * k_nyquist);
}

/// Integer mode numbers kept per axis: everything strictly below Nyquist.
[[nodiscard]] inline int max_mode(int N) noexcept { return N / 2 - 1; }

/**
 * @brief A grid on which the map's mode sum is well sampled for this `f`.
 *
 * The map is \f$N\f$-independent (pinned by the unit test) provided the
 * content edge \f$k_c = fN/2\f$ falls on enough integer modes to resolve
 * the Gaussian spectrum; below \f$k_c\approx5\f$ it does not. This picks
 * \f$N\f$ so that \f$k_c \approx 12\f$ whatever `f` is, which keeps a
 * bisection over `f` both correct at small `f` and cheap at large `f`.
 */
[[nodiscard]] inline int auto_map_grid(double f) {
  const int N = 2 * static_cast<int>(std::ceil(12.0 / f));
  return std::max(N, 16);
}

/**
 * @brief Semi-analytic \f$L^2\f$ error of the FD path, relative to the
 *        initial RMS, for a field whose content reaches fraction `f` of
 *        Nyquist.
 *
 * Exact for the linear heat equation (see the file header): each mode is
 * evolved by its own operator eigenvalue and the errors are combined by
 * Parseval. The spectral path's value is identically zero and is not
 * computed here.
 *
 * @param fd_order Even FD order 2..20.
 * @param f        Content fraction of Nyquist, \f$0 < f \le 1\f$.
 * @param N        Grid points per axis for the mode cube.
 * @param tau      \f$D\,t\,k_c^2\f$.
 */
[[nodiscard]] inline double predict_l2_error(int fd_order, double f, int N = kMapGrid,
                                             double tau = kDiffusionTime) {
  if (!(f > 0.0) || f > 1.0)
    throw std::invalid_argument("predict_l2_error: need 0 < f <= 1");
  if (N < 8 || (N % 2) != 0)
    throw std::invalid_argument("predict_l2_error: need an even N >= 8");

  const double k_c = f * static_cast<double>(N) / 2.0; // k_Nyquist = N/2 at L=2pi
  const double shape = content_shape_constant();
  // Two truncations, both harmless: Nyquist (the grid cannot carry more)
  // and the point where the Gaussian weight has fallen to ~1e-35, which
  // bounds the neglected tail's contribution to the reported error at
  // ~1e-17 -- below anything this study quotes.
  const int m_max = std::min(max_mode(N),
                             static_cast<int>(std::ceil(2.4 * k_c)) + 1);
  // Per-axis tables over n = -m_max .. m_max (symmetric, so store n >= 0).
  std::vector<double> weight(static_cast<std::size_t>(m_max) + 1);
  std::vector<double> nu_sq(static_cast<std::size_t>(m_max) + 1);
  std::vector<double> defect(static_cast<std::size_t>(m_max) + 1);
  for (int n = 0; n <= m_max; ++n) {
    const double nu = static_cast<double>(n) / k_c;
    const double theta = M_PI * f * nu; // = 2*pi*n/N, i.e. k*dx
    weight[static_cast<std::size_t>(n)] = std::exp(-shape * nu * nu);
    nu_sq[static_cast<std::size_t>(n)] = nu * nu;
    // Per-axis contribution to Delta = t * (lambda_p - (-k^2)), made
    // dimensionless: t/dx^2 = tau/(pi^2 f^2).
    defect[static_cast<std::size_t>(n)] =
        fd_symbol_defect(fd_order, theta) * tau / (M_PI * M_PI * f * f);
  }

  double norm = 0.0;
  double sum_sq = 0.0;
  for (int i = -m_max; i <= m_max; ++i) {
    const std::size_t ai = static_cast<std::size_t>(std::abs(i));
    for (int j = -m_max; j <= m_max; ++j) {
      const std::size_t aj = static_cast<std::size_t>(std::abs(j));
      const double w_ij = weight[ai] * weight[aj];
      for (int k = -m_max; k <= m_max; ++k) {
        const std::size_t ak = static_cast<std::size_t>(std::abs(k));
        const double w = w_ij * weight[ak];
        norm += w;
        const double decay = std::exp(-tau * (nu_sq[ai] + nu_sq[aj] + nu_sq[ak]));
        // exp(Delta) - 1 via expm1: Delta is tiny for a high order and a
        // fine grid, and this is the only place precision could be lost.
        const double diff =
            decay * std::expm1(defect[ai] + defect[aj] + defect[ak]);
        sum_sq += w * diff * diff;
      }
    }
  }
  return std::sqrt(sum_sq / norm);
}

/**
 * @brief Largest content fraction `f` at which FD order `fd_order` still
 *        meets an \f$L^2\f$ target `eps` — the coarsest grid it may use.
 *
 * Monotone in `f` (a coarser grid is never more accurate), so a bisection
 * is safe. Returns 1.0 when the target is met even with content at Nyquist,
 * and `f_min` when it is unreachable across the bracket.
 */
[[nodiscard]] inline double content_fraction_at(int fd_order, double eps,
                                                double tau = kDiffusionTime,
                                                double f_min = 1.0e-3) {
  const auto err = [&](double f) {
    return predict_l2_error(fd_order, f, auto_map_grid(f), tau);
  };
  double lo = f_min, hi = 1.0;
  if (err(hi) <= eps) return hi;
  if (err(lo) > eps) return lo;
  for (int it = 0; it < 40; ++it) {
    const double mid = 0.5 * (lo + hi);
    if (err(mid) <= eps) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

/**
 * @brief Content fraction above which an FD order stops being the cheaper
 *        route to a fixed accuracy: \f$\sqrt[3]{c_\mathrm{fd}/c_\mathrm{spec}}\f$.
 *
 * Pure cost arithmetic under the \f$N^3\f$ assumption stated in the file
 * header — no numerics enter. This is the crossover predicate the report
 * quotes and `tests/test_heat3d_spectral_content.cpp` pins.
 */
[[nodiscard]] inline double crossover_fraction(double cost_fd,
                                               double cost_spectral) {
  if (!(cost_fd > 0.0) || !(cost_spectral > 0.0))
    throw std::invalid_argument("crossover_fraction: costs must be positive");
  return std::cbrt(cost_fd / cost_spectral);
}

/**
 * @brief Equal-accuracy cost of an FD order relative to the spectral path
 *        (spectral = 1), given the coarsest grid it may use.
 *
 * @param f_star         Output of `content_fraction_at()`.
 * @param cost_fd        Measured FD per-step cost at a reference grid.
 * @param cost_spectral  Measured spectral per-step cost at the same grid.
 */
[[nodiscard]] inline double equal_accuracy_cost_ratio(double f_star, double cost_fd,
                                                      double cost_spectral) {
  if (!(f_star > 0.0)) throw std::invalid_argument("equal_accuracy_cost_ratio: f>0");
  return (cost_fd / (f_star * f_star * f_star)) / cost_spectral;
}

/// True when FD order with cost `cost_fd` beats the spectral path at the
/// accuracy whose tolerated content fraction is `f_star`.
[[nodiscard]] inline bool fd_cheaper_than_spectral(double f_star, double cost_fd,
                                                   double cost_spectral) {
  return f_star > crossover_fraction(cost_fd, cost_spectral);
}

// ---------------------------------------------------------------------------
// Validation against real runs of the production FD stack
// ---------------------------------------------------------------------------

/**
 * @brief One axis factor of the truncated-Gaussian field, evolved exactly.
 *
 * \f$g(x,t) = \hat a_0 + 2\sum_{n=1}^{M}\hat a_n e^{-D n^2 t}\cos(nx)\f$
 * with \f$\hat a_n = e^{-\sigma^2 n^2/2}\f$. At \f$t=0\f$ this is the
 * initial condition; at \f$t>0\f$ it is the *exact* solution of the PDE for
 * that initial condition, because the field is a finite Fourier sum and the
 * heat equation is diagonal in that basis. The 3-D field is the product of
 * three such factors on the three coordinates, and so is its exact
 * evolution — which is why no FFT is needed anywhere in this study.
 */
[[nodiscard]] inline double gaussian_axis_factor(double x, double t, double sigma,
                                                 int m_max) noexcept {
  double g = 1.0; // n = 0 term, a_0 = 1
  for (int n = 1; n <= m_max; ++n) {
    const double nn = static_cast<double>(n);
    const double amp = std::exp(-0.5 * sigma * sigma * nn * nn);
    if (amp < 1.0e-300) break;
    g += 2.0 * amp * std::exp(-kD * nn * nn * t) * std::cos(nn * x);
  }
  return g;
}

/// One validation point: what the map predicted, and what a run measured.
struct ValidationCase {
  int fd_order{2};
  int N{64};
  double f{0.3};
  double tau{kDiffusionTime};
  int n_steps{0};
  double dt{0.0};
  double t_final{0.0};
  /// `predict_l2_error(fd_order, f, N, tau)`.
  double predicted_l2{0.0};
  /// RMS of (RK4-stepped FD field - exact solution) over RMS of the IC.
  double measured_l2{0.0};
  /// `measured_l2 / predicted_l2`; 1 means the closed form is right.
  double ratio{0.0};
};

/**
 * @brief Run the real FD stack on the controlled-content field and measure
 *        its \f$L^2\f$ error against the exact solution.
 *
 * Uses exactly the objects `heat3d_fd` uses — a padded
 * `pfc::data::Field`, `pfc::comm::HaloExchange`, and
 * `pfc::gradient::FDGradient<HeatGrads>` — so what is validated is the
 * shipped operator, not a reimplementation of its symbol.
 *
 * **Time integration is classical RK4, not the driver's forward Euler.**
 * The quantity under test is a *spatial* error as small as \f$10^{-9}\f$;
 * forward Euler's \f$O(\Delta t)\f$ error would swamp it unless
 * \f$\Delta t\f$ were absurd. RK4's \f$O(\Delta t^4)\f$ falls below the
 * spatial floor at a few hundred steps, and the driver reports a
 * step-halved value so the reader can see that it has.
 *
 * Single rank by construction, like `convergence_study.hpp`.
 *
 * @param fd_order Even FD order 2..20.
 * @param N        Grid points per axis.
 * @param f        Content fraction of Nyquist.
 * @param tau      \f$D\,t\,k_c^2\f$.
 * @param n_steps  RK4 steps to `t_final`.
 */
[[nodiscard]] inline ValidationCase run_validation(int fd_order, int N, double f,
                                                   double tau = kDiffusionTime,
                                                   int n_steps = 400) {
  ValidationCase result;
  result.fd_order = fd_order;
  result.N = N;
  result.f = f;
  result.tau = tau;
  result.n_steps = n_steps;

  const double dx = kDomainLength / static_cast<double>(N);
  const double k_c = f * static_cast<double>(N) / 2.0;
  const double sigma = content_sigma(f, N);
  const int m_max = max_mode(N);
  const double t_final = tau / (kD * k_c * k_c);
  const double dt = t_final / static_cast<double>(n_steps);
  result.dt = dt;
  result.t_final = t_final;

  const auto domain =
      pfc::domain::create(pfc::GridSize({N, N, N}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                          pfc::GridSpacing({dx, dx, dx}));
  const auto decomp = pfc::decomposition::create(domain, /*nproc=*/1);

  const int hw = fd_order / 2;
  auto make = [&] {
    return pfc::data::field_from_subdomain<double>(decomp, /*rank=*/0, hw);
  };
  pfc::data::Field<double, pfc::HostSpace> u = make();
  pfc::data::Field<double, pfc::HostSpace> w = make();
  pfc::data::Field<double, pfc::HostSpace> k1 = make();
  pfc::data::Field<double, pfc::HostSpace> k2 = make();
  pfc::data::Field<double, pfc::HostSpace> k3 = make();
  pfc::data::Field<double, pfc::HostSpace> k4 = make();

  pfc::comm::HaloExchange<pfc::HostSpace, double> halo(w, decomp, /*rank=*/0,
                                                       MPI_COMM_WORLD);
  pfc::gradient::FDGradient<HeatGrads> grad(w, fd_order);

  const auto ic = [&](double x, double y, double z) {
    return gaussian_axis_factor(x, 0.0, sigma, m_max) *
           gaussian_axis_factor(y, 0.0, sigma, m_max) *
           gaussian_axis_factor(z, 0.0, sigma, m_max);
  };
  u.apply(ic);

  double ic_sq = 0.0, cells = 0.0;
  u.for_each_owned([&](int i, int j, int k) {
    ic_sq += u(i, j, k) * u(i, j, k);
    cells += 1.0;
  });
  const double ic_rms = std::sqrt(ic_sq / cells);

  // Laplacian of `w` into `out`, halo exchanged first.
  const auto laplacian = [&](pfc::data::Field<double, pfc::HostSpace> &out) {
    halo.exchange();
    out.for_each_owned([&](int i, int j, int k) {
      const auto g = pfc::gradient::evaluate(grad, pfc::Int3{i, j, k});
      out(i, j, k) = kD * (g.xx + g.yy + g.zz);
    });
  };
  const auto blend = [&](const pfc::data::Field<double, pfc::HostSpace> &stage,
                         double factor) {
    w.for_each_owned([&](int i, int j, int k) {
      w(i, j, k) = u(i, j, k) + factor * stage(i, j, k);
    });
  };

  for (int step = 0; step < n_steps; ++step) {
    w.for_each_owned([&](int i, int j, int k) { w(i, j, k) = u(i, j, k); });
    laplacian(k1);
    blend(k1, 0.5 * dt);
    laplacian(k2);
    blend(k2, 0.5 * dt);
    laplacian(k3);
    blend(k3, dt);
    laplacian(k4);
    u.for_each_owned([&](int i, int j, int k) {
      u(i, j, k) += (dt / 6.0) * (k1(i, j, k) + 2.0 * k2(i, j, k) +
                                  2.0 * k3(i, j, k) + k4(i, j, k));
    });
  }

  // Exact solution of the same sampled field, from the separable series.
  std::vector<double> axis(static_cast<std::size_t>(N));
  for (int i = 0; i < N; ++i) {
    axis[static_cast<std::size_t>(i)] =
        gaussian_axis_factor(static_cast<double>(i) * dx, t_final, sigma, m_max);
  }
  double err_sq = 0.0;
  u.for_each_owned([&](int i, int j, int k) {
    const double exact = axis[static_cast<std::size_t>(i)] *
                         axis[static_cast<std::size_t>(j)] *
                         axis[static_cast<std::size_t>(k)];
    const double d = u(i, j, k) - exact;
    err_sq += d * d;
  });

  result.measured_l2 = std::sqrt(err_sq / cells) / ic_rms;
  result.predicted_l2 = predict_l2_error(fd_order, f, N, tau);
  result.ratio =
      (result.predicted_l2 > 0.0) ? result.measured_l2 / result.predicted_l2 : 0.0;
  return result;
}

} // namespace heat3d::spectral_content
