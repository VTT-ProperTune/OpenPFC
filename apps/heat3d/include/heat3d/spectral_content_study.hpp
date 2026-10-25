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
 * Spectral-content study for the heat equation application.
 * measures the spectral and
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
 * numerics. With the measured costs in `out/report/data/heat3d_method_cost.csv`
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
 * @see openpfc/kernel/field/fd_symbols.hpp — stencil symbols and defect.
 * @see openpfc/kernel/field/periodic_spectra.hpp — occupancy-matched
 *      families and Parseval error used by this study.
 * @see apps/tungsten/include/tungsten/resolution.hpp — the *nonlinear*
 *      half of the same question (aliasing of a cubic term), measured
 *      separately; nothing here says anything about it.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
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
#include <openpfc/kernel/field/fd_symbols.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/field/periodic_spectra.hpp>

#include <heat3d/heat_model.hpp>

namespace heat3d::spectral_content {

/// Periodic box length on every axis, so mode numbers are plain integers.
inline constexpr double kDomainLength = 2.0 * M_PI;

/// Amplitude ratio defining the "content edge" \f$k_c\f$ (see file header).
inline constexpr double kContentThreshold = pfc::field::spectra::kContentThreshold;

/// Dimensionless diffusion time \f$\tau = D\,t\,k_c^2\f$ used everywhere.
inline constexpr double kDiffusionTime = pfc::field::spectra::kDiffusionTime;

/// Frozen `|measured/predicted - 1|` bound for `--held-out-families`.
/// Do not retune: job 22085944 was admitted against this number.
inline constexpr double kFamilyHeldOutRatioTol = 1.0e-6;

/// True when the held-out family RK4 residual is inside the frozen bound.
[[nodiscard]] inline bool family_heldout_passes(double max_abs_rel) noexcept {
  return max_abs_rel < kFamilyHeldOutRatioTol;
}

/// Throw after diagnostics have been written if the residual violates
/// @ref kFamilyHeldOutRatioTol. Printing `FAIL` and exiting 0 is not a
/// gate.
inline void require_family_heldout_ok(double max_abs_rel) {
  if (family_heldout_passes(max_abs_rel)) {
    return;
  }
  throw std::runtime_error(
      "held-out family RK4 residual exceeds frozen |ratio-1| bound " +
      std::to_string(kFamilyHeldOutRatioTol) +
      " (max |ratio-1| = " + std::to_string(max_abs_rel) + ")");
}

/// Grid used for the reported semi-analytic sweep. The map is
/// \f$N\f$-independent above ~48 (pinned by the unit test); 128 is chosen
/// so even the smallest \f$f\f$ in the sweep still has a well-sampled
/// spectrum.
inline constexpr int kMapGrid = pfc::field::spectra::kMapGrid;

/// \f$\sigma^2 k_c^2 = 2\ln(1/\text{threshold})\f$: the one constant tying
/// the Gaussian width to the content fraction.
[[nodiscard]] inline double content_shape_constant() noexcept {
  return 2.0 * std::log(1.0 / kContentThreshold);
}

[[nodiscard]] inline double arcsin_series_coefficient(int m) {
  return pfc::field::fd::arcsin_series_coefficient(m);
}

[[nodiscard]] inline double fd_symbol(int order, double theta) {
  return pfc::field::fd::d2_symbol(order, theta);
}

[[nodiscard]] inline double fd_symbol_defect(int order, double theta) {
  return pfc::field::fd::d2_symbol_defect(order, theta);
}

[[nodiscard]] inline double content_sigma(double f, int N) {
  const double k_nyquist = M_PI / (kDomainLength / static_cast<double>(N));
  return std::sqrt(content_shape_constant()) / (f * k_nyquist);
}

[[nodiscard]] inline int max_mode(int N) noexcept { return N / 2 - 1; }

[[nodiscard]] inline int auto_map_grid(double f) {
  return pfc::field::spectra::auto_map_grid(f);
}

[[nodiscard]] inline double predict_l2_error(int fd_order, double f,
                                             int N = kMapGrid,
                                             double tau = kDiffusionTime) {
  return pfc::field::spectra::predict_heat_l2_error(pfc::field::spectra::gaussian(),
                                                    fd_order, f, N, tau, 3);
}

[[nodiscard]] inline pfc::field::spectra::ContentFractionMatch
content_fraction_at(int fd_order, double eps, double tau = kDiffusionTime,
                    double f_min = 1.0e-3) {
  return pfc::field::spectra::content_fraction_at(pfc::field::spectra::gaussian(),
                                                  fd_order, eps, tau, 3, f_min);
}

[[nodiscard]] inline double crossover_fraction(double cost_fd,
                                               double cost_spectral) {
  return pfc::field::spectra::crossover_fraction(cost_fd, cost_spectral);
}

[[nodiscard]] inline double equal_accuracy_cost_ratio(double f_star, double cost_fd,
                                                      double cost_spectral) {
  return pfc::field::spectra::equal_accuracy_cost_ratio(f_star, cost_fd,
                                                        cost_spectral);
}

[[nodiscard]] inline bool fd_cheaper_than_spectral(double f_star, double cost_fd,
                                                   double cost_spectral) {
  return pfc::field::spectra::fd_cheaper_than_spectral(f_star, cost_fd,
                                                       cost_spectral);
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
  const char *family_id{"gaussian"};
  int fd_order{2};
  int N{64};
  double f{0.3};
  double tau{kDiffusionTime};
  int n_steps{0};
  double dt{0.0};
  double t_final{0.0};
  /// Parseval prediction for this family.
  double predicted_l2{0.0};
  /// RMS of (RK4-stepped FD field - exact solution) over RMS of the IC.
  double measured_l2{0.0};
  /// `measured_l2 / predicted_l2`; 1 means the closed form is right.
  double ratio{0.0};
};

/**
 * @brief Run the real FD stack on a periodic spectrum family and measure
 *        its \f$L^2\f$ error against the exact solution.
 *
 * Uses exactly the objects `heat3d_fd` uses — a padded
 * `pfc::data::Field`, `pfc::comm::HaloExchange`, and
 * `pfc::gradient::FDGradient<HeatGrads>` — so what is validated is the
 * shipped operator, not a reimplementation of its symbol.
 *
 * The Gaussian family uses the separable cosine product (fast). Other
 * families use `evaluate_periodic_field` and should be run on modest `N`.
 */
[[nodiscard]] inline ValidationCase run_validation(
    int fd_order, int N, double f, double tau = kDiffusionTime, int n_steps = 400,
    pfc::field::spectra::SpectrumFamily fam = pfc::field::spectra::gaussian()) {
  ValidationCase result;
  result.family_id = fam.id;
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
  const bool gaussian_fast = std::string_view(fam.id) == "gaussian";

  const auto domain = pfc::domain::create(pfc::GridSize({N, N, N}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
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
    if (gaussian_fast) {
      return gaussian_axis_factor(x, 0.0, sigma, m_max) *
             gaussian_axis_factor(y, 0.0, sigma, m_max) *
             gaussian_axis_factor(z, 0.0, sigma, m_max);
    }
    return pfc::field::spectra::evaluate_periodic_field(fam, x, y, z, 0.0, f, N, 3,
                                                        kD);
  };
  u.apply(ic);

  double ic_sq = 0.0, cells = 0.0;
  u.for_each_owned([&](int i, int j, int k) {
    ic_sq += u(i, j, k) * u(i, j, k);
    cells += 1.0;
  });
  const double ic_rms = std::sqrt(ic_sq / cells);

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

  std::vector<double> axis;
  if (gaussian_fast) {
    axis.resize(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
      axis[static_cast<std::size_t>(i)] =
          gaussian_axis_factor(static_cast<double>(i) * dx, t_final, sigma, m_max);
    }
  }
  double err_sq = 0.0;
  u.for_each_owned([&](int i, int j, int k) {
    double exact = 0.0;
    if (gaussian_fast) {
      exact = axis[static_cast<std::size_t>(i)] * axis[static_cast<std::size_t>(j)] *
              axis[static_cast<std::size_t>(k)];
    } else {
      exact = pfc::field::spectra::evaluate_periodic_field(
          fam, static_cast<double>(i) * dx, static_cast<double>(j) * dx,
          static_cast<double>(k) * dx, t_final, f, N, 3, kD);
    }
    const double d = u(i, j, k) - exact;
    err_sq += d * d;
  });

  result.measured_l2 = std::sqrt(err_sq / cells) / ic_rms;
  result.predicted_l2 =
      pfc::field::spectra::predict_heat_l2_error(fam, fd_order, f, N, tau, 3);
  result.ratio =
      (result.predicted_l2 > 0.0) ? result.measured_l2 / result.predicted_l2 : 0.0;
  return result;
}

} // namespace heat3d::spectral_content
