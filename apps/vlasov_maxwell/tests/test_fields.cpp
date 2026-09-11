// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_fields.cpp
 * @brief Oracles for the field side of the 1D2V Vlasov-Maxwell application:
 *        the Gauss solve, the exactly integrated light wave, the Duhamel
 *        source terms, the velocity-space quadratures, and the Gauss residual.
 *
 * @details
 * Every test here has an oracle that is not the code. Three kinds of them, and
 * the distinction matters for reading the tolerances:
 *
 *  1. **Round-off oracles.** A closed-form solution the discrete operator
 *     reproduces *exactly*, so the only error is floating point. The vacuum
 *     electromagnetic wave, the Poisson solve on resolved modes, and the
 *     charge-conserving Ampere step are all of this kind. If one of them
 *     fails, something is wrong with the mathematics, not with the
 *     resolution.
 *
 *     "Round-off" is not one number, and the tolerances here are set from the
 *     measured floor of each operator rather than from a preference:
 *       - a *field* compared against its closed form is good to ~1e-15 on a
 *         power-of-two grid, so those assertions are at `1e-13`;
 *       - a *Gauss residual* is `max|d_x E_x - rho| / max|rho|`, and the
 *         spectral derivative multiplies the round-off of `E_x` by `k_max`,
 *         so its floor is `eps k_max |E_x| / |rho|` -- measured at `1e-14`
 *         to `1e-13` on these grids. Those assertions are at `1e-12`;
 *       - a non-power-of-two grid falls back to the direct `O(N^2)` DFT,
 *         whose error grows like `N` rather than `log N`. The standing-wave
 *         test (`nx = 20`, 300 steps) measures `3.0e-13` where the
 *         power-of-two path gives `1e-15`, so it is asserted at `1e-12`.
 *
 *  2. **A measured convergence oracle.** The velocity quadrature is
 *     spectrally accurate only while `f` is negligible on the velocity
 *     boundary, so its error is predicted analytically -- `2 erfc(v_max/(sqrt2
 *     v_th))` from tail truncation -- and the test asserts the *prediction*,
 *     not merely smallness. It deliberately includes a truncating box where
 *     the error is `5e-3` and reports it, because choosing a box where the
 *     error vanishes and then claiming the quadrature is exact would be
 *     circular.
 *
 *  3. **A discrimination oracle.** The Gauss residual is asserted to be at
 *     round-off with a charge-conserving current *and* to be visibly larger
 *     with a midpoint current that is second-order accurate but not
 *     charge-conserving. A diagnostic that is small for both is not measuring
 *     what the issue claims it measures.
 *
 * There is deliberately **no test of `div B = 0`.** `B_x == 0` by construction
 * in 1D; asserting it would report a property of the geometry as evidence
 * about the code, which is worse than reporting nothing.
 *
 * @see maxwell.hpp for the derivations these tests check
 * @see moments.hpp for why midpoint quadrature is the right rule here
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <vector>

#include <catch2/catch_all.hpp>
#include <mpi.h>

#include <vlasov_maxwell/ics.hpp>
#include <vlasov_maxwell/maxwell.hpp>
#include <vlasov_maxwell/moments.hpp>
#include <vlasov_maxwell/parameters.hpp>
#include <vlasov_maxwell/phase_space.hpp>
#include <vlasov_maxwell/step.hpp>

using Catch::Approx;
using vlasov::Complex;
using vlasov::SimParams;
using vlasov::SpectralLine1D;

namespace {

constexpr double kPi = 3.14159265358979323846;

int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

int world_rank() {
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}

/// `max |a - b|` over two equally sized fields.
double max_diff(const std::vector<double> &a, const std::vector<double> &b) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

double max_abs(const std::vector<double> &a) {
  double m = 0.0;
  for (double v : a) m = std::max(m, std::fabs(v));
  return m;
}

/// Cell-centred sampling of `g(x)` on the `x` axis of `p`.
template <class G> std::vector<double> sample_x(const SimParams &p, G &&g) {
  std::vector<double> u(static_cast<std::size_t>(p.nx));
  for (int i = 0; i < p.nx; ++i) u[static_cast<std::size_t>(i)] = g(p.x_of(i));
  return u;
}

/**
 * @brief A drifting Maxwellian on the whole phase-space grid.
 *
 *     f(x, v) = [n0 + dn cos(k0 x)] * exp(-|v - u|^2 / 2 v_th^2) / (2 pi v_th^2)
 *
 * so that, *if the velocity box were infinite*,
 *   `int f dv = n(x)`, `int v f dv = n(x) u`,
 *   `int |v|^2 f dv = n(x) (|u|^2 + 2 v_th^2)`.
 * Those are the oracles; the finite box is the error under study.
 *
 * Layout: `x` slowest, then `v_x`, `v_y` fastest -- the layout
 * `StridedDistribution::contiguous` assumes and the one that makes the `v_y`
 * halo contiguous.
 */
std::vector<double> maxwellian(const SimParams &p, double n0, double dn,
                               double ux, double uy, double vth) {
  const std::size_t n = static_cast<std::size_t>(p.nx) *
                        static_cast<std::size_t>(p.nvx) *
                        static_cast<std::size_t>(p.nvy);
  std::vector<double> f(n);
  const double norm = 1.0 / (2.0 * kPi * vth * vth);
  const double inv2s2 = 1.0 / (2.0 * vth * vth);
  const double k0 = p.k0();
  for (int i = 0; i < p.nx; ++i) {
    const double dens = n0 + dn * std::cos(k0 * p.x_of(i));
    for (int j = 0; j < p.nvx; ++j) {
      const double vx = p.vx_of(j) - ux;
      for (int k = 0; k < p.nvy; ++k) {
        const double vy = p.vy_of(k) - uy;
        const std::size_t idx =
            (static_cast<std::size_t>(i) * static_cast<std::size_t>(p.nvx) +
             static_cast<std::size_t>(j)) *
                static_cast<std::size_t>(p.nvy) +
            static_cast<std::size_t>(k);
        f[idx] = dens * norm * std::exp(-(vx * vx + vy * vy) * inv2s2);
      }
    }
  }
  return f;
}

/// View over this rank's slab of a *full* phase-space array, decomposed on
/// `v_y`. Used to check that the `MPI_Allreduce` in `reduce_velocity` is the
/// only thing standing between a slab and the right answer.
vlasov::StridedDistribution slab_view(const std::vector<double> &full,
                                      const SimParams &p, int kbegin, int kend) {
  vlasov::StridedDistribution v;
  v.data = full.data() + kbegin;
  v.nx = p.nx;
  v.nvx = p.nvx;
  v.kbegin = kbegin;
  v.kend = kend;
  v.stride_vy = 1;
  v.stride_vx = p.nvy;
  v.stride_x = static_cast<std::ptrdiff_t>(p.nvy) * p.nvx;
  return v;
}

} // namespace

// ===========================================================================
// 1. Poisson / Gauss
// ===========================================================================

/*
 * Oracle. For rho(x) = A cos(k x) with k = 2 pi m / Lx, the unique zero-mean
 * periodic solution of d_x E_x = rho is
 *
 *     E_x(x) = (A/k) sin(k x),
 *
 * because d_x[(A/k) sin(kx)] = A cos(kx) and sin has zero mean. A spectral
 * solve represents both functions exactly on the grid -- a single resolved
 * Fourier mode is not approximated by a DFT, it *is* a DFT coefficient -- so
 * the only error is floating point.
 */
TEST_CASE("Poisson reproduces a single-mode analytic field to round-off",
          "[unit][poisson]") {
  SimParams p;
  p.nx = 64;
  p.Lx = 3.0;
  SpectralLine1D line(p.nx, p.Lx);

  for (int m : {1, 2, 5, 13}) {
    const double k = p.k_skin(m);
    const double A = 0.7;
    const auto rho = sample_x(p, [&](double x) { return A * std::cos(k * x); });
    const auto exact = sample_x(p, [&](double x) { return (A / k) * std::sin(k * x); });

    const auto r = vlasov::solve_gauss(line, rho);
    INFO("mode m = " << m << " err = " << max_diff(r.Ex, exact));
    REQUIRE(r.neutral);
    REQUIRE(std::fabs(r.net_charge) < 1e-15);
    REQUIRE(max_diff(r.Ex, exact) < 1e-14);

    // And the constraint it was built to satisfy. The residual differentiates
    // E_x spectrally, which multiplies its round-off by k; at m = 13 on this
    // box k = 27 and the measured floor is 1.2e-14.
    const auto g = vlasov::gauss_residual(line, r.Ex, rho);
    INFO("mode m = " << m << " Gauss residual = " << g.residual);
    REQUIRE(g.residual < 1e-12);
  }
}

/*
 * Oracle. Poisson is linear, so a sum of modes maps to the sum of the
 * single-mode solutions:
 *
 *     rho = sum_m A_m cos(k_m x + phi_m)
 *     E_x = sum_m (A_m/k_m) sin(k_m x + phi_m).
 *
 * This is a stronger statement than the single-mode test: it fails if the
 * wavenumber indexing is wrong for any mode -- in particular if the negative
 * half of the spectrum (m > nx/2) is given the wrong sign, which a
 * single-mode test with m = 1 cannot see.
 */
TEST_CASE("Poisson is linear over a multi-mode charge density", "[unit][poisson]") {
  SimParams p;
  p.nx = 128;
  p.Lx = 2.0 * kPi;
  SpectralLine1D line(p.nx, p.Lx);

  const std::array<int, 5> modes{1, 3, 7, 20, 61};
  const std::array<double, 5> amps{1.0, -0.4, 0.25, 0.11, -0.03};
  const std::array<double, 5> phase{0.0, 0.9, 2.1, -1.3, 0.4};

  auto rho_fn = [&](double x) {
    double s = 0.0;
    for (std::size_t q = 0; q < modes.size(); ++q) {
      s += amps[q] * std::cos(p.k_skin(modes[q]) * x + phase[q]);
    }
    return s;
  };
  auto exact_fn = [&](double x) {
    double s = 0.0;
    for (std::size_t q = 0; q < modes.size(); ++q) {
      const double k = p.k_skin(modes[q]);
      s += (amps[q] / k) * std::sin(k * x + phase[q]);
    }
    return s;
  };

  const auto rho = sample_x(p, rho_fn);
  const auto exact = sample_x(p, exact_fn);
  const auto r = vlasov::solve_gauss(line, rho);
  const double gres = vlasov::gauss_residual(line, r.Ex, rho).residual;
  INFO("multi-mode err = " << max_diff(r.Ex, exact) << ", Gauss residual = " << gres);
  REQUIRE(max_diff(r.Ex, exact) < 1e-14);
  REQUIRE(gres < 1e-12);
}

/*
 * Oracle. Integrating d_x E_x = rho over the period gives 0 = int rho dx: a
 * periodic electrostatic field can only exist for a neutral plasma. So for
 * rho = A cos(kx) + C with C != 0 there is *no* solution, and the honest
 * behaviour is to solve the neutralised problem and say so. The test asserts
 * that the leftover charge is reported exactly (it is the mean, C) and that
 * the returned field solves the neutralised equation, i.e. its divergence
 * matches rho - C and therefore misses rho by exactly C.
 */
TEST_CASE("a non-neutral charge density is reported, not silently absorbed",
          "[unit][poisson]") {
  SimParams p;
  p.nx = 32;
  p.Lx = 1.5;
  SpectralLine1D line(p.nx, p.Lx);

  const double k = p.k_skin(1);
  const double A = 0.4;
  const double C = 0.05;
  const auto rho = sample_x(p, [&](double x) { return A * std::cos(k * x) + C; });

  const auto r = vlasov::solve_gauss(line, rho);
  REQUIRE_FALSE(r.neutral);
  REQUIRE(r.net_charge == Approx(C).epsilon(1e-13));

  // The field solves the neutralised problem exactly...
  const auto neutral_rho = sample_x(p, [&](double x) { return A * std::cos(k * x); });
  REQUIRE(vlasov::gauss_residual(line, r.Ex, neutral_rho).residual < 1e-12);
  // ...and therefore misses the real rho by exactly the net charge.
  const auto g = vlasov::gauss_residual(line, r.Ex, rho);
  REQUIRE(g.abs_residual == Approx(std::fabs(C)).epsilon(1e-12));
  REQUIRE(g.net_charge == Approx(C).epsilon(1e-13));

  // And a call site that wants the run to stop can have that.
  REQUIRE_THROWS_AS(vlasov::require_neutral(r), std::runtime_error);

  // A neutral density must not throw.
  REQUIRE_NOTHROW(vlasov::require_neutral(vlasov::solve_gauss(line, neutral_rho)));
}

// ===========================================================================
// 2. The vacuum electromagnetic wave -- validation stage 1
// ===========================================================================

/*
 * Oracle. With J = 0 the transverse pair is
 *     d_t E_y = -d_x B_z,   d_t B_z = -d_x E_y.
 * Substituting E_y = F(x - t), B_z = F(x - t) satisfies both for any F:
 *     d_t E_y = -F' = -d_x B_z,   d_t B_z = -F' = -d_x E_y.
 * So E_y = B_z = A cos(k x) is a *right*-travelling wave of speed exactly 1
 * (= c in these units) and at time t the field is A cos(k(x - t)).
 *
 * The point of this test is the phrase "for any dt". The update is the matrix
 * exponential exp(-i k dt S), which is the exact solution operator, so a
 * single step of dt = 37.3 -- nearly six wave periods at k = 1 -- must be as
 * accurate as a thousand steps of dt = 0.0373. A scheme that merely
 * approximated the light wave would fail the large-dt column catastrophically
 * and pass the small-dt one, which is what makes this the test that proves
 * exactness rather than accuracy.
 */
TEST_CASE("the vacuum light wave propagates at c = 1 exactly, at any dt",
          "[unit][maxwell][stage1]") {
  SimParams p;
  p.nx = 64;
  p.Lx = 2.0 * kPi;
  SpectralLine1D line(p.nx, p.Lx);

  const int m = 3;
  const double k = p.k_skin(m);
  const double A = 0.85;
  const double t_end = 37.3;

  for (int nsteps : {1, 2, 7, 1000}) {
    const double dt = t_end / static_cast<double>(nsteps);
    auto Ey = sample_x(p, [&](double x) { return A * std::cos(k * x); });
    auto Bz = Ey;
    for (int n = 0; n < nsteps; ++n) {
      vlasov::advance_transverse_vacuum(line, Ey, Bz, dt);
    }
    const auto exact =
        sample_x(p, [&](double x) { return A * std::cos(k * (x - t_end)); });
    const double e_err = max_diff(Ey, exact);
    const double b_err = max_diff(Bz, exact);
    INFO("nsteps = " << nsteps << " dt = " << dt << " Ey err = " << e_err
                     << " Bz err = " << b_err);
    REQUIRE(e_err < 1e-13);
    REQUIRE(b_err < 1e-13);
  }

  // The left-travelling branch, which differs only by the sign of B_z:
  // E_y = F(x + t), B_z = -F(x + t).
  {
    auto Ey = sample_x(p, [&](double x) { return A * std::cos(k * x); });
    auto Bz = sample_x(p, [&](double x) { return -A * std::cos(k * x); });
    vlasov::advance_transverse_vacuum(line, Ey, Bz, t_end);
    const auto exact =
        sample_x(p, [&](double x) { return A * std::cos(k * (x + t_end)); });
    INFO("left-mover err = " << max_diff(Ey, exact));
    REQUIRE(max_diff(Ey, exact) < 1e-13);
  }
}

/*
 * Oracle. exp(-i k dt S) is unitary for real k (S is real symmetric, so
 * -i k dt S is anti-Hermitian), hence it preserves |Ey_hat|^2 + |Bz_hat|^2
 * mode by mode; Parseval turns that into exact conservation of
 * (1/2) int (E_y^2 + B_z^2) dx. Round-off only, for arbitrary multi-mode data
 * and for as many steps as one cares to take.
 */
TEST_CASE("vacuum field energy is conserved to round-off", "[unit][maxwell][stage1]") {
  SimParams p;
  p.nx = 128;
  p.Lx = 4.0;
  SpectralLine1D line(p.nx, p.Lx);

  std::mt19937 rng(20260911u);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  std::vector<double> Ey(static_cast<std::size_t>(p.nx));
  std::vector<double> Bz(static_cast<std::size_t>(p.nx));
  // Band-limited random data: a few modes, so nothing sits on Nyquist.
  for (int mm = 1; mm <= 12; ++mm) {
    const double k = p.k_skin(mm);
    const double a = uni(rng);
    const double b = uni(rng);
    const double pa = uni(rng);
    const double pb = uni(rng);
    for (int i = 0; i < p.nx; ++i) {
      const double x = p.x_of(i);
      Ey[static_cast<std::size_t>(i)] += a * std::cos(k * x + pa);
      Bz[static_cast<std::size_t>(i)] += b * std::cos(k * x + pb);
    }
  }

  const double e0 = vlasov::transverse_energy(line, Ey, Bz);
  REQUIRE(e0 > 0.1);
  double worst = 0.0;
  const double dt = 0.137;
  for (int n = 0; n < 500; ++n) {
    vlasov::advance_transverse_vacuum(line, Ey, Bz, dt);
    worst = std::max(worst, std::fabs(vlasov::transverse_energy(line, Ey, Bz) / e0 - 1.0));
  }
  INFO("500 steps, worst relative energy drift = " << worst);
  REQUIRE(worst < 1e-13);
}

/*
 * Oracle. Superposing the right- and left-travelling solutions of amplitude
 * A/2 gives the standing wave
 *
 *     E_y(x,t) = A cos(k x) cos(k t),    B_z(x,t) = A sin(k x) sin(k t),
 *
 * from E_y(0) = A cos(kx), B_z(0) = 0. Its E_y nodes are fixed in space at
 * k x = pi/2 + n pi and stay there for all time; the wave oscillates in place
 * rather than moving. A scheme with any dispersion error would drift them.
 *
 * The grid is chosen so that four of those nodes land exactly on cell centres:
 * with nx = 20, m = 2 and cell centres x_i = (i + 1/2) dx,
 *     k x_i = 2 pi * 2 * (i + 1/2)/20 = pi (i + 1/2)/5,
 * which is pi/2 modulo pi at i = 2, 7, 12, 17. nx = 20 is not a power of two,
 * so this test also exercises the direct-DFT fallback in `maxwell.hpp`.
 */
TEST_CASE("a standing wave holds its analytic node positions",
          "[unit][maxwell][stage1]") {
  SimParams p;
  p.nx = 20;
  p.Lx = 2.0 * kPi;
  SpectralLine1D line(p.nx, p.Lx);

  const int m = 2;
  const double k = p.k_skin(m);
  const double A = 1.3;
  auto Ey = sample_x(p, [&](double x) { return A * std::cos(k * x); });
  std::vector<double> Bz(static_cast<std::size_t>(p.nx), 0.0);

  const double dt = 0.11;
  double t = 0.0;
  double node_worst = 0.0;
  double form_worst = 0.0;
  for (int n = 0; n < 300; ++n) {
    vlasov::advance_transverse_vacuum(line, Ey, Bz, dt);
    t += dt;
    const auto ey_exact =
        sample_x(p, [&](double x) { return A * std::cos(k * x) * std::cos(k * t); });
    const auto bz_exact =
        sample_x(p, [&](double x) { return A * std::sin(k * x) * std::sin(k * t); });
    form_worst = std::max({form_worst, max_diff(Ey, ey_exact), max_diff(Bz, bz_exact)});
    for (int i : {2, 7, 12, 17}) {
      node_worst = std::max(node_worst, std::fabs(Ey[static_cast<std::size_t>(i)]));
    }
  }
  // nx = 20 is not a power of two, so this runs on the direct O(N^2) DFT
  // whose round-off grows like N; the measured floor over 300 steps is 3.0e-13
  // against an amplitude of 1.3, i.e. 2e-13 relative.
  INFO("standing wave: worst form error " << form_worst << ", worst node value "
                                          << node_worst);
  REQUIRE(form_worst < 1e-12);
  REQUIRE(node_worst < 1e-13);
}

/*
 * Oracle. omega(k) = k, for every resolved k, exactly.
 *
 * Measurement: initialise the pure right-travelling mode m (E_y = B_z =
 * cos(k x)), whose Fourier coefficient at index m evolves as exp(-i omega t)
 * with omega = k. Advance by t_m = (pi/2)/k, so the phase advance is exactly
 * pi/2 for every mode and cannot wrap, then read
 *     omega_measured = -arg( hat_m(t) / hat_m(0) ) / t_m.
 *
 * The Nyquist index is excluded and this is not a loophole: on an even grid
 * d_x of the Nyquist mode vanishes at every grid point, so the mode carries no
 * representable derivative and no wave. That is a property of the sampling,
 * stated in `maxwell.hpp`, and pretending otherwise would be the actual error.
 */
TEST_CASE("numerical dispersion is omega = k to 1e-12 for every resolved mode",
          "[unit][maxwell][stage1]") {
  SimParams p;
  p.nx = 64;
  p.Lx = 5.0;
  SpectralLine1D line(p.nx, p.Lx);

  double worst = 0.0;
  int worst_m = 0;
  for (int m = 1; m < p.nx / 2; ++m) {
    const double k = p.k_skin(m);
    const double t = 0.5 * kPi / k;
    auto Ey = sample_x(p, [&](double x) { return std::cos(k * x); });
    auto Bz = Ey;
    const auto h0 = line.forward(Ey);
    vlasov::advance_transverse_vacuum(line, Ey, Bz, t);
    const auto h1 = line.forward(Ey);

    const Complex ratio = h1[static_cast<std::size_t>(m)] / h0[static_cast<std::size_t>(m)];
    const double omega = -std::arg(ratio) / t;
    const double rel = std::fabs(omega - k) / k;
    if (rel > worst) {
      worst = rel;
      worst_m = m;
    }
    // The amplitude must not change either: the propagator is a rotation.
    REQUIRE(std::fabs(std::abs(ratio) - 1.0) < 1e-13);
  }
  INFO("worst relative dispersion error " << worst << " at mode " << worst_m
                                          << " of " << p.nx / 2 - 1 << " resolved");
  REQUIRE(worst < 1e-12);
}

// ===========================================================================
// 3. The Duhamel (ETD) source terms
// ===========================================================================

/*
 * Oracle for ETD1. With a current constant in time the transverse system
 *     dy/dt = -i k S y + b,   b = (-Jy_hat, 0)
 * has the closed-form solution
 *     y(t) = exp(-i k t S) (y0 - y*) + y*,   y* = (i k S)^{-1} b = -i S b / k,
 * verified by substitution (S^2 = I). This is an independent solution, not the
 * update formula, so agreement to round-off is a real check on M1 -- and it
 * must hold for *any* number of steps, since ETD1 is exact when the source is
 * constant.
 */
TEST_CASE("ETD1 is exact for a time-independent current", "[unit][maxwell][etd]") {
  SimParams p;
  p.nx = 32;
  p.Lx = 2.0 * kPi;
  SpectralLine1D line(p.nx, p.Lx);

  const int m = 2;
  const double k = p.k_skin(m);
  const double J0 = 0.31;
  const auto Jy = sample_x(p, [&](double x) { return J0 * std::cos(k * x); });
  auto Ey = sample_x(p, [&](double x) { return 0.2 * std::sin(k * x); });
  auto Bz = sample_x(p, [&](double x) { return -0.1 * std::cos(k * x); });

  const auto ey0 = line.forward(Ey);
  const auto bz0 = line.forward(Bz);
  const auto jh = line.forward(Jy);

  const double t_end = 4.7;
  const int nsteps = 11;
  const double dt = t_end / nsteps;
  for (int n = 0; n < nsteps; ++n) {
    vlasov::advance_transverse_etd1(line, Ey, Bz, Jy, dt);
  }

  // Build the analytic answer mode by mode.
  std::vector<Complex> ey_ref(static_cast<std::size_t>(p.nx));
  std::vector<Complex> bz_ref(static_cast<std::size_t>(p.nx));
  for (int q = 0; q < p.nx; ++q) {
    const std::size_t qq = static_cast<std::size_t>(q);
    const double kq = line.k_deriv(q);
    const Complex J = jh[qq];
    Complex es{0.0, 0.0};
    Complex bs{0.0, 0.0};
    if (kq != 0.0) {
      // y* = -i S b / k with b = (-J, 0) gives (0, +i J / k).
      bs = Complex{0.0, 1.0} * (J / kq);
    } else {
      REQUIRE(std::abs(J) < 1e-13); // no k=0 current in this test
    }
    const double th = kq * t_end;
    const Complex de = ey0[qq] - es;
    const Complex db = bz0[qq] - bs;
    ey_ref[qq] = std::cos(th) * de - Complex{0.0, 1.0} * (std::sin(th) * db) + es;
    bz_ref[qq] = std::cos(th) * db - Complex{0.0, 1.0} * (std::sin(th) * de) + bs;
  }
  const auto ey_exact = line.inverse(ey_ref);
  const auto bz_exact = line.inverse(bz_ref);
  INFO("ETD1 constant-source err Ey " << max_diff(Ey, ey_exact) << " Bz "
                                      << max_diff(Bz, bz_exact));
  REQUIRE(max_diff(Ey, ey_exact) < 1e-13);
  REQUIRE(max_diff(Bz, bz_exact) < 1e-13);
}

/*
 * Oracle for ETD2. With a current linear in time, b(t) = b0 + b1 t, the
 * particular solution of dy/dt = -i k S y + b(t) is y_p = p + q t with
 *     q = -i S b1 / k,        p = -i S (b0 - q) / k,
 * (match powers of t; S^2 = I), so
 *     y(t) = exp(-i k t S) (y0 - p) + p + q t.
 * ETD2 extrapolates the source linearly from b_{n-1} and b_n, which is *exact*
 * for a linear source, so agreement must again be at round-off -- and ETD1 on
 * the same problem must not be, or the second-order term is doing nothing.
 */
TEST_CASE("ETD2 is exact for a current linear in time, and ETD1 is not",
          "[unit][maxwell][etd]") {
  SimParams p;
  p.nx = 32;
  p.Lx = 2.0 * kPi;
  SpectralLine1D line(p.nx, p.Lx);

  const int m = 2;
  const double k = p.k_skin(m);
  const double J0 = 0.25;
  const double rate = 0.6; // Jy(x,t) = J0 cos(kx) (1 + rate t)
  auto jy_at = [&](double t) {
    return sample_x(p, [&](double x) { return J0 * std::cos(k * x) * (1.0 + rate * t); });
  };

  const double t_end = 3.0;
  const int nsteps = 12;
  const double dt = t_end / nsteps;

  auto Ey0 = sample_x(p, [&](double x) { return 0.15 * std::sin(k * x); });
  auto Bz0 = sample_x(p, [&](double x) { return 0.05 * std::cos(k * x); });

  auto Ey = Ey0;
  auto Bz = Bz0;
  for (int n = 0; n < nsteps; ++n) {
    const double tn = n * dt;
    vlasov::advance_transverse_etd2(line, Ey, Bz, jy_at(tn), jy_at(tn - dt), dt);
  }

  auto Ey1 = Ey0;
  auto Bz1 = Bz0;
  for (int n = 0; n < nsteps; ++n) {
    vlasov::advance_transverse_etd1(line, Ey1, Bz1, jy_at(n * dt), dt);
  }

  // Analytic reference.
  const auto eh0 = line.forward(Ey0);
  const auto bh0 = line.forward(Bz0);
  const auto j0h = line.forward(jy_at(0.0));
  std::vector<Complex> ey_ref(static_cast<std::size_t>(p.nx));
  std::vector<Complex> bz_ref(static_cast<std::size_t>(p.nx));
  for (int q = 0; q < p.nx; ++q) {
    const std::size_t qq = static_cast<std::size_t>(q);
    const double kq = line.k_deriv(q);
    const Complex J0h = j0h[qq];          // b0 = (-J0h, 0)
    const Complex J1h = rate * j0h[qq];   // b1 = (-J1h, 0)
    Complex qe{0.0, 0.0}, qb{0.0, 0.0}, pe{0.0, 0.0}, pb{0.0, 0.0};
    if (kq != 0.0) {
      // -i S (a, 0) / k = -i (0, a) / k; with a = -J this is (0, +i J / k).
      qb = Complex{0.0, 1.0} * (J1h / kq);
      // p = -i S (b0 - q) / k; b0 - q = (-J0h - qe, 0 - qb) = (-J0h, -qb).
      const Complex c1 = -J0h - qe;
      const Complex c2 = Complex{0.0, 0.0} - qb;
      pe = Complex{0.0, -1.0} * (c2 / kq);
      pb = Complex{0.0, -1.0} * (c1 / kq);
    } else {
      REQUIRE(std::abs(J0h) < 1e-13);
    }
    const double th = kq * t_end;
    const Complex de = eh0[qq] - pe;
    const Complex db = bh0[qq] - pb;
    ey_ref[qq] = std::cos(th) * de - Complex{0.0, 1.0} * (std::sin(th) * db) + pe +
                 qe * t_end;
    bz_ref[qq] = std::cos(th) * db - Complex{0.0, 1.0} * (std::sin(th) * de) + pb +
                 qb * t_end;
  }
  const auto ey_exact = line.inverse(ey_ref);
  const auto bz_exact = line.inverse(bz_ref);

  const double err2 = std::max(max_diff(Ey, ey_exact), max_diff(Bz, bz_exact));
  const double err1 = std::max(max_diff(Ey1, ey_exact), max_diff(Bz1, bz_exact));
  INFO("ETD2 err " << err2 << ", ETD1 err " << err1);
  REQUIRE(err2 < 1e-13);
  REQUIRE(err1 > 1e-3); // the first-order term is not free
}

/*
 * Oracle. The measured order of ETD1 and ETD2 against a current that is
 * neither constant nor linear. Halving dt must divide the error by 2 and 4
 * respectively; the fitted slopes are reported.
 */
TEST_CASE("the Duhamel terms achieve their stated orders", "[unit][maxwell][etd]") {
  SimParams p;
  p.nx = 32;
  p.Lx = 2.0 * kPi;
  SpectralLine1D line(p.nx, p.Lx);

  const double k = p.k_skin(2);
  const double omega = 1.7;
  auto jy_at = [&](double t) {
    return sample_x(p, [&](double x) { return 0.4 * std::cos(k * x) * std::cos(omega * t); });
  };
  const double t_end = 1.0;

  auto run = [&](int nsteps, bool second) {
    const double dt = t_end / nsteps;
    std::vector<double> Ey(static_cast<std::size_t>(p.nx), 0.0);
    std::vector<double> Bz(static_cast<std::size_t>(p.nx), 0.0);
    for (int n = 0; n < nsteps; ++n) {
      const double tn = n * dt;
      if (second) {
        vlasov::advance_transverse_etd2(line, Ey, Bz, jy_at(tn), jy_at(tn - dt), dt);
      } else {
        vlasov::advance_transverse_etd1(line, Ey, Bz, jy_at(tn), dt);
      }
    }
    return Ey;
  };

  const auto ref = run(20480, true); // dt = 5e-5, well below the errors compared
  for (bool second : {false, true}) {
    std::vector<double> errs;
    for (int n : {40, 80, 160, 320}) errs.push_back(max_diff(run(n, second), ref));
    const double s1 = std::log2(errs[0] / errs[1]);
    const double s2 = std::log2(errs[1] / errs[2]);
    const double s3 = std::log2(errs[2] / errs[3]);
    INFO((second ? "ETD2" : "ETD1") << " errors " << errs[0] << " " << errs[1] << " "
                                    << errs[2] << " " << errs[3] << " slopes " << s1
                                    << " " << s2 << " " << s3);
    const double want = second ? 2.0 : 1.0;
    REQUIRE(s3 == Approx(want).margin(0.15));
  }
}

// ===========================================================================
// 4. Charge conservation and the Gauss residual
// ===========================================================================

/*
 * Oracle -- the identity the whole application turns on.
 *
 * Take a charge density and current that satisfy continuity exactly:
 *     rho(x,t) = A cos(k x - w t),     J_x(x,t) = (A w / k) cos(k x - w t),
 * since d_t rho + d_x J_x = A w sin(.) - A w sin(.) = 0.
 *
 * Discretely, Ampere advances E_x by the *step-averaged* current
 *     Jbar_n = (1/dt) int_{t_n}^{t_n+dt} J_x dt'
 *            = -(A/(k dt)) [ sin(k x - w t_{n+1}) - sin(k x - w t_n) ],
 * for which the discrete continuity relation
 *     (rho_{n+1} - rho_n)/dt + d_x Jbar_n = 0
 * holds to round-off. Then
 *     E_x^{n+1} = E_x^n - dt Jbar_n = E_x^n + (A/k)[sin(kx - w t_{n+1}) - sin(kx - w t_n)]
 * telescopes to E_x^n = (A/k) sin(k x - w t_n) from E_x^0 = (A/k) sin(k x), and
 * d_x E_x^n = rho_n exactly. **Gauss is preserved by charge conservation, not
 * by the field solver**, which is why the residual is the diagnostic it is.
 *
 * The contrast half of the test replaces Jbar with the midpoint-in-time
 * current, which is second-order accurate and *not* charge-conserving; the
 * residual must then be visibly nonzero. A diagnostic that stayed at round-off
 * for both would be measuring nothing.
 */
TEST_CASE("Gauss stays at round-off exactly when the current conserves charge",
          "[unit][maxwell][gauss]") {
  SimParams p;
  p.nx = 64;
  p.Lx = 2.0 * kPi;
  SpectralLine1D line(p.nx, p.Lx);

  const double k = p.k_skin(1);
  const double A = 0.3;
  const double w = 0.7;
  const double dt = 0.25;
  const int nsteps = 40;

  auto rho_at = [&](double t) {
    return sample_x(p, [&](double x) { return A * std::cos(k * x - w * t); });
  };
  auto jbar = [&](double t) {
    return sample_x(p, [&](double x) {
      return -(A / (k * dt)) *
             (std::sin(k * x - w * (t + dt)) - std::sin(k * x - w * t));
    });
  };
  auto jmid = [&](double t) {
    return sample_x(p, [&](double x) {
      return (A * w / k) * std::cos(k * x - w * (t + 0.5 * dt));
    });
  };

  // The construction really does satisfy discrete continuity.
  {
    double worst = 0.0;
    for (int n = 0; n < nsteps; ++n) {
      const double t = n * dt;
      const auto r0 = rho_at(t);
      const auto r1 = rho_at(t + dt);
      const auto dj = line.derivative(jbar(t));
      for (int i = 0; i < p.nx; ++i) {
        const std::size_t ii = static_cast<std::size_t>(i);
        worst = std::max(worst, std::fabs((r1[ii] - r0[ii]) / dt + dj[ii]));
      }
    }
    INFO("discrete continuity residual " << worst);
    REQUIRE(worst < 1e-13);
  }

  auto Ex = sample_x(p, [&](double x) { return (A / k) * std::sin(k * x); });
  const double g0 = vlasov::gauss_residual(line, Ex, rho_at(0.0)).residual;
  INFO("initial Gauss residual " << g0);
  REQUIRE(g0 < 1e-12);

  double worst_cons = 0.0;
  for (int n = 0; n < nsteps; ++n) {
    const double t = n * dt;
    vlasov::ampere_ex(Ex, jbar(t), dt);
    worst_cons = std::max(worst_cons,
                          vlasov::gauss_residual(line, Ex, rho_at(t + dt)).residual);
  }
  INFO("charge-conserving current: worst Gauss residual " << worst_cons);
  REQUIRE(worst_cons < 1e-12);

  // The same run with a second-order-accurate but non-conserving current.
  auto Ex2 = sample_x(p, [&](double x) { return (A / k) * std::sin(k * x); });
  double worst_mid = 0.0;
  for (int n = 0; n < nsteps; ++n) {
    const double t = n * dt;
    vlasov::ampere_ex(Ex2, jmid(t), dt);
    worst_mid = std::max(worst_mid,
                         vlasov::gauss_residual(line, Ex2, rho_at(t + dt)).residual);
  }
  INFO("midpoint current: worst Gauss residual " << worst_mid
                                                 << " (dt = " << dt << ")");
  REQUIRE(worst_mid > 1e-6);
  REQUIRE(worst_mid / worst_cons > 1e5);
}

/*
 * Oracle. The optional divergence correction must (a) report the residual it
 * is about to hide, and (b) actually remove it, leaving the mean of E_x --
 * which Gauss says nothing about -- untouched.
 */
TEST_CASE("the divergence correction reports the residual it removes",
          "[unit][maxwell][gauss]") {
  SimParams p;
  p.nx = 32;
  p.Lx = 3.0;
  SpectralLine1D line(p.nx, p.Lx);

  const double k = p.k_skin(1);
  const auto rho = sample_x(p, [&](double x) { return 0.5 * std::cos(k * x); });
  const double offset = 0.42;
  // A field that is wrong by a visible amount plus a constant offset.
  auto Ex = sample_x(p, [&](double x) {
    return (0.5 / k) * std::sin(k * x) * 0.8 + offset;
  });

  const double before = vlasov::gauss_residual(line, Ex, rho).residual;
  REQUIRE(before == Approx(0.2).epsilon(1e-10)); // 20% of rho is unmatched

  // With apply = false nothing changes and the number is still reported.
  const auto probe = vlasov::correct_divergence(line, Ex, rho, false);
  REQUIRE_FALSE(probe.corrected);
  REQUIRE(probe.residual == Approx(before));

  const auto g = vlasov::correct_divergence(line, Ex, rho, true);
  REQUIRE(g.corrected);
  REQUIRE(g.residual == Approx(before)); // the UNcorrected residual, as promised
  REQUIRE(vlasov::gauss_residual(line, Ex, rho).residual < 1e-12);
  REQUIRE(line.mean(Ex) == Approx(offset).epsilon(1e-12));
}

// ===========================================================================
// 5. Velocity-space moments
// ===========================================================================

/*
 * Oracle. For the drifting Maxwellian in `maxwellian()` with an effectively
 * infinite velocity box,
 *     n(x)          = n0 + dn cos(k0 x)
 *     int v_x f dv  = n(x) u_x,       int v_y f dv = n(x) u_y
 *     int |v|^2 f dv = n(x) (|u|^2 + 2 v_th^2)
 * and with sigma = -1 and a neutralising background chosen for exact
 * neutrality,
 *     rho(x) = -dn cos(k0 x),   J_x(x) = -n(x) u_x,   J_y(x) = -n(x) u_y.
 *
 * The box here is v_max = 10 v_th, whose truncation error is
 * 2 erfc(10/sqrt2) ~ 1e-23, i.e. below round-off; the next test is the one
 * that studies the error rather than hiding from it.
 */
TEST_CASE("a drifting Maxwellian deposits the analytic rho and J",
          "[unit][moments]") {
  SimParams p;
  p.nx = 16;
  p.nvx = 128;
  p.nvy = 128;
  p.Lx = 4.0;
  const double vth = 0.05;
  const double ux = 0.1;
  const double uy = -0.04;
  p.v_max = 10.0 * vth;

  const double n0 = 1.0;
  const double dn = 0.3;
  const auto f = maxwellian(p, n0, dn, ux, uy, vth);
  const auto view = vlasov::StridedDistribution::contiguous(f.data(), p.nx, p.nvx, 0, p.nvy);

  vlasov::ReductionOptions opt;
  opt.v_thermal = vth;
  // This test builds the *whole* velocity grid on every rank, so the v_y
  // "decomposition" is trivial and the reduction must be over MPI_COMM_SELF.
  // Completing it over MPI_COMM_WORLD would multiply every moment by the rank
  // count -- which is exactly what the decomposition test below checks does
  // *not* happen when the slabs really are disjoint.
  opt.comm = MPI_COMM_SELF;
  const vlasov::Species s{"electron", -1.0, 1.0};
  vlasov::VelocityMoments mom;
  const auto src = vlasov::deposit(p, s, view, mom, opt);

  const double k0 = p.k0();
  const auto n_exact = sample_x(p, [&](double x) { return n0 + dn * std::cos(k0 * x); });
  const auto rho_exact = sample_x(p, [&](double x) { return -dn * std::cos(k0 * x); });
  const auto jx_exact =
      sample_x(p, [&](double x) { return -(n0 + dn * std::cos(k0 * x)) * ux; });
  const auto jy_exact =
      sample_x(p, [&](double x) { return -(n0 + dn * std::cos(k0 * x)) * uy; });

  INFO("n err " << max_diff(mom.n, n_exact) << ", rho err " << max_diff(src.rho, rho_exact)
                << ", Jx err " << max_diff(src.Jx, jx_exact) << ", Jy err "
                << max_diff(src.Jy, jy_exact));
  REQUIRE(max_diff(mom.n, n_exact) < 1e-13);
  REQUIRE(max_diff(src.rho, rho_exact) < 1e-13);
  REQUIRE(max_diff(src.Jx, jx_exact) < 1e-13);
  REQUIRE(max_diff(src.Jy, jy_exact) < 1e-13);

  // The background chosen for exact neutrality is +n0, and the state is neutral.
  REQUIRE(src.background == Approx(n0).epsilon(1e-13));
  REQUIRE(std::fabs(src.net_charge) < 1e-14);

  // Scalars. int n dx = n0 Lx (the cosine integrates to zero over the period).
  REQUIRE(mom.number == Approx(n0 * p.Lx).epsilon(1e-12));
  // Kinetic energy = (mu/2) int n dx (|u|^2 + 2 v_th^2).
  const double w_exact = 0.5 * s.mu * n0 * p.Lx * (ux * ux + uy * uy + 2.0 * vth * vth);
  REQUIRE(vlasov::kinetic_energy(p, s, mom) == Approx(w_exact).epsilon(1e-12));
  // Momentum = mu int n dx u.
  const auto pm = vlasov::momentum(p, s, mom);
  REQUIRE(pm[0] == Approx(s.mu * n0 * p.Lx * ux).epsilon(1e-12));
  REQUIRE(pm[1] == Approx(s.mu * n0 * p.Lx * uy).epsilon(1e-12));

  // Casimirs. f is positive, so L1 = int f = the particle number.
  REQUIRE(mom.l1 == Approx(mom.number).epsilon(1e-12));
  REQUIRE(mom.f_min > 0.0);

  // And the velocity boundary is unoccupied, which is what licenses all of
  // the above. The box is 10 v_th but the drift is 2 v_th, so the nearest face
  // is 8 v_th from the peak and the oracle is exp(-8^2/2) = 1.3e-14 -- measured
  // 2.4e-14, the rest being the v_y offset of the drift. The shell within one
  // thermal width of a face is then 7 v_th out: ~erfc(7/sqrt2) = 8e-13.
  INFO("face ratio " << mom.face_ratio() << ", boundary fraction "
                     << mom.boundary_fraction);
  REQUIRE(mom.face_ratio() < 1e-13);
  REQUIRE(mom.face_ratio() > 1e-15); // it is a measurement, not a zero
  REQUIRE(mom.boundary_fraction < 1e-11);
}

/*
 * Oracle. Midpoint quadrature of a Gaussian on a uniform grid is spectrally
 * accurate; the residual error is dominated by *truncation* of the tail
 * outside the box. For each velocity axis the omitted mass is
 *     eps = erfc(v_max / (sqrt2 v_th)),
 * and the 2-D integrand factorises, so the relative deposition error is
 *     (1 - eps)^2 subtracted from 1, i.e.  2 eps - eps^2.
 *
 * The grid spacing is held at dv = 0.1 v_th throughout, so the aliasing error
 * exp(-2 pi^2 v_th^2/dv^2) = exp(-1974) is exactly zero and truncation is the
 * whole story.
 *
 * The point of the test is not that the error is small -- at v_max = 3 v_th it
 * is 5e-3, which is reported -- but that it is *the predicted one*, and that
 * it falls to round-off by v_max = 8 v_th.
 */
TEST_CASE("the velocity quadrature error is the predicted truncation error",
          "[unit][moments]") {
  const double vth = 0.1;
  const double n0 = 1.0;
  const double dn = 0.5;

  std::vector<double> measured;
  std::vector<double> predicted;
  for (int mult : {3, 4, 5, 6, 7, 8}) {
    SimParams p;
    p.nx = 8;
    p.Lx = 2.0;
    p.v_max = static_cast<double>(mult) * vth;
    p.nvx = 20 * mult; // dv = 2 v_max / nvx = 0.1 v_th
    p.nvy = 20 * mult;
    REQUIRE(p.dvx() == Approx(0.1 * vth).epsilon(1e-12));

    const auto f = maxwellian(p, n0, dn, 0.0, 0.0, vth);
    const auto view =
        vlasov::StridedDistribution::contiguous(f.data(), p.nx, p.nvx, 0, p.nvy);
    vlasov::ReductionOptions opt;
    opt.comm = MPI_COMM_SELF; // whole velocity grid held locally; see above
    const auto mom = vlasov::reduce_velocity(p, view, opt);

    const double k0 = p.k0();
    const auto n_exact = sample_x(p, [&](double x) { return n0 + dn * std::cos(k0 * x); });
    const double err = max_diff(mom.n, n_exact) / max_abs(n_exact);

    const double eps = std::erfc(static_cast<double>(mult) / std::sqrt(2.0));
    const double pred = 2.0 * eps - eps * eps;
    measured.push_back(err);
    predicted.push_back(pred);
    INFO("v_max = " << mult << " v_th : measured " << err << ", predicted " << pred);
    // Above round-off, the prediction is the oracle.
    if (pred > 1e-12) {
      REQUIRE(err == Approx(pred).epsilon(0.2));
    }
  }

  // Report, then assert. The truncating box really is bad...
  INFO("errors: " << measured[0] << " " << measured[1] << " " << measured[2] << " "
                  << measured[3] << " " << measured[4] << " " << measured[5]);
  REQUIRE(measured.front() > 1e-3);
  // ...the error falls monotonically...
  for (std::size_t q = 1; q < measured.size(); ++q) {
    REQUIRE(measured[q] < measured[q - 1]);
  }
  // ...and reaches round-off, which is the regime a valid run must be in.
  REQUIRE(measured.back() < 1e-13);
}

/*
 * Oracle. The velocity-boundary occupancy must *detect* the truncation the
 * previous test measures. In a box of v_max = 3 v_th the outermost cell centre
 * sits at 2.95 v_th, so
 *     max|f| on a face / max|f| = exp(-2.95^2/2) = 1.3e-2,
 * and the particle number within one thermal width of a boundary
 * (|v_x| > 2 v_th or |v_y| > 2 v_th) is, by the union of two axis events,
 *     ~ 2 erfc(2/sqrt2) - erfc(2/sqrt2)^2 = 8.9e-2.
 * Both must be large; in the v_max = 10 v_th box of the test above both are
 * below 1e-15. The diagnostic is only worth reporting if it separates those.
 */
TEST_CASE("the velocity-boundary occupancy detects a truncating box",
          "[unit][moments]") {
  const double vth = 0.1;
  SimParams p;
  p.nx = 4;
  p.Lx = 1.0;
  p.v_max = 3.0 * vth;
  p.nvx = 60;
  p.nvy = 60;

  const auto f = maxwellian(p, 1.0, 0.0, 0.0, 0.0, vth);
  const auto view =
      vlasov::StridedDistribution::contiguous(f.data(), p.nx, p.nvx, 0, p.nvy);
  vlasov::ReductionOptions opt;
  opt.v_thermal = vth;
  opt.comm = MPI_COMM_SELF; // whole velocity grid held locally; see above
  const auto mom = vlasov::reduce_velocity(p, view, opt);

  const double outer = p.v_max - 0.5 * p.dvx(); // 2.95 v_th
  const double face_pred = std::exp(-0.5 * (outer / vth) * (outer / vth));
  const double eps = std::erfc(2.0 / std::sqrt(2.0));
  const double frac_pred = 2.0 * eps - eps * eps;

  INFO("face ratio " << mom.face_ratio() << " (predicted " << face_pred
                     << "), boundary fraction " << mom.boundary_fraction
                     << " (predicted " << frac_pred << ")");
  REQUIRE(mom.face_ratio() == Approx(face_pred).epsilon(0.02));
  // The union estimate ignores the doubly-counted corner, so it overshoots by
  // a few percent; measured 8.38e-2 against a predicted 8.89e-2.
  REQUIRE(mom.boundary_fraction == Approx(frac_pred).epsilon(0.10));
  REQUIRE(mom.face_ratio() > 1e-2);
  REQUIRE(mom.boundary_fraction > 1e-2);
}

/*
 * Oracle. The reduction is a sum over the decomposed v_y axis, so splitting
 * the axis across ranks and completing with MPI_Allreduce must give bitwise
 * the same partial-sum structure only up to the order of accumulation -- i.e.
 * the same answer to round-off, and the *same* answer on every rank. This runs
 * under any rank count; on one rank it degenerates to the serial path and
 * still checks the analytic oracle.
 */
TEST_CASE("the moment reduction is correct under a v_y decomposition",
          "[unit][moments][mpi]") {
  const int nproc = world_size();
  const int rank = world_rank();

  SimParams p;
  p.nx = 12;
  p.nvx = 64;
  p.nvy = 64;
  p.Lx = 3.0;
  const double vth = 0.06;
  const double ux = 0.05;
  p.v_max = 12.0 * vth;
  if (p.nvy % nproc != 0) {
    // The split below assumes an even division; the decomposition itself is
    // not restricted this way, only this test's construction of it.
    SUCCEED("rank count " << nproc << " does not divide nvy; split not exercised");
    return;
  }

  const double n0 = 1.0;
  const double dn = 0.2;
  const auto f = maxwellian(p, n0, dn, ux, 0.0, vth);

  const int per = p.nvy / nproc;
  const auto view = slab_view(f, p, rank * per, (rank + 1) * per);
  const auto mom = vlasov::reduce_velocity(p, view);

  const double k0 = p.k0();
  const auto n_exact = sample_x(p, [&](double x) { return n0 + dn * std::cos(k0 * x); });
  const auto jx_exact =
      sample_x(p, [&](double x) { return (n0 + dn * std::cos(k0 * x)) * ux; });
  INFO("nproc = " << nproc << " n err " << max_diff(mom.n, n_exact) << " flux err "
                  << max_diff(mom.flux_x, jx_exact));
  REQUIRE(max_diff(mom.n, n_exact) < 1e-13);
  REQUIRE(max_diff(mom.flux_x, jx_exact) < 1e-13);
  REQUIRE(max_abs(mom.flux_y) < 1e-15);
  REQUIRE(mom.number == Approx(n0 * p.Lx).epsilon(1e-12));

  // Every rank must hold the identical replicated result, or the field solve
  // silently diverges between ranks.
  std::vector<double> gathered = mom.n;
  MPI_Bcast(gathered.data(), static_cast<int>(gathered.size()), MPI_DOUBLE, 0,
            MPI_COMM_WORLD);
  REQUIRE(max_diff(gathered, mom.n) == 0.0);
}

/*
 * Oracle. Charge is linear in the species, so two species of opposite sign and
 * equal density must deposit exactly zero charge and, if they counter-stream,
 * twice the single-species current. This is the algebra `add_species`
 * implements and the configuration every two-stream and Weibel benchmark uses.
 */
TEST_CASE("multi-species deposition superposes", "[unit][moments]") {
  SimParams p;
  p.nx = 8;
  p.nvx = 96;
  p.nvy = 96;
  p.Lx = 2.0;
  const double vth = 0.05;
  const double u = 0.15;
  p.v_max = 8.0 * vth + u;
  p.rho_background = 0.0; // explicit: the species already neutralise each other

  const auto fp = maxwellian(p, 1.0, 0.0, +u, 0.0, vth);
  const auto fm = maxwellian(p, 1.0, 0.0, -u, 0.0, vth);
  const auto vp = vlasov::StridedDistribution::contiguous(fp.data(), p.nx, p.nvx, 0, p.nvy);
  const auto vm = vlasov::StridedDistribution::contiguous(fm.data(), p.nx, p.nvx, 0, p.nvy);

  const vlasov::Species electron{"electron", -1.0, 1.0};
  const vlasov::Species ion{"ion", +1.0, vlasov::kProtonElectronMassRatio};

  vlasov::ReductionOptions opt;
  opt.comm = MPI_COMM_SELF; // whole velocity grid held locally; see above
  auto src = vlasov::Sources::zeros(p.nx);
  const auto me = vlasov::reduce_velocity(p, vp, opt);
  const auto mi = vlasov::reduce_velocity(p, vm, opt);
  vlasov::add_species(electron, me, src);
  vlasov::add_species(ion, mi, src);
  vlasov::apply_background(p, src);

  INFO("net charge " << src.net_charge << " max|rho| " << max_abs(src.rho)
                     << " max|Jx| " << max_abs(src.Jx));
  REQUIRE(max_abs(src.rho) < 1e-13);
  REQUIRE(std::fabs(src.net_charge) < 1e-14);
  REQUIRE(src.background == 0.0);
  // J_x = (-1)(+u) + (+1)(-u) = -2u per unit density.
  for (double j : src.Jx) REQUIRE(j == Approx(-2.0 * u).epsilon(1e-12));

  // The ion mass enters the kinetic energy and the momentum, not the charge.
  const auto pe = vlasov::momentum(p, electron, me);
  const auto pi = vlasov::momentum(p, ion, mi);
  REQUIRE(pe[0] == Approx(p.Lx * u).epsilon(1e-12));
  REQUIRE(pi[0] == Approx(-vlasov::kProtonElectronMassRatio * p.Lx * u).epsilon(1e-12));
}

// ===========================================================================
// 6. Field energy bookkeeping
// ===========================================================================

/*
 * Oracle. For E_x = a cos(kx), E_y = b sin(kx), B_z = c cos(kx) on a period
 * Lx, Parseval gives (1/2) int (E_x^2 + E_y^2 + B_z^2) dx = (Lx/4)(a^2+b^2+c^2),
 * and int E_y B_z dx = 0 because sin and cos are orthogonal over the period.
 * Shifting B_z to c sin(kx) makes the momentum (Lx/2) b c.
 */
TEST_CASE("field energy and momentum match their Parseval values",
          "[unit][maxwell][diagnostics]") {
  SimParams p;
  p.nx = 64;
  p.Lx = 2.5;
  SpectralLine1D line(p.nx, p.Lx);
  const double k = p.k_skin(2);
  const double a = 0.3;
  const double b = -0.7;
  const double c = 0.11;

  vlasov::FieldState f;
  f.Ex = sample_x(p, [&](double x) { return a * std::cos(k * x); });
  f.Ey = sample_x(p, [&](double x) { return b * std::sin(k * x); });
  f.Bz = sample_x(p, [&](double x) { return c * std::cos(k * x); });

  REQUIRE(vlasov::field_energy(line, f) ==
          Approx(0.25 * p.Lx * (a * a + b * b + c * c)).epsilon(1e-12));
  REQUIRE(vlasov::transverse_energy(line, f.Ey, f.Bz) ==
          Approx(0.25 * p.Lx * (b * b + c * c)).epsilon(1e-12));
  REQUIRE(std::fabs(vlasov::field_momentum_x(line, f)) < 1e-14);

  f.Bz = sample_x(p, [&](double x) { return c * std::sin(k * x); });
  REQUIRE(vlasov::field_momentum_x(line, f) == Approx(0.5 * p.Lx * b * c).epsilon(1e-12));
}

/*
 * Oracle. For a uniform 2-D Maxwellian of density n0 and thermal width vth,
 *
 *     S = -int f ln f d x d^{2}v
 *       = - n0 Lx [ ln(n0 / (2 pi vth^{2})) - 1 ]
 *
 * because int f ln f d^{2}v = n [ln(n/(2 pi vth^{2})) - 1] over the infinite
 * velocity plane. The box here is 10 vth, whose omitted tail is below
 * round-off, so the discrete -sum f ln f * dV has to land on that closed
 * form. Entropy is the Casimir that numerical diffusion shows up in, and
 * it was previously only compared host-vs-device, never against an answer.
 */
TEST_CASE("entropy of a Maxwellian matches -int f ln f",
          "[unit][moments][entropy]") {
  SimParams p;
  p.nx = 8;
  p.nvx = 128;
  p.nvy = 128;
  p.Lx = 2.0;
  const double vth = 0.05;
  p.v_max = 10.0 * vth;
  const double n0 = 1.0;
  const auto f = maxwellian(p, n0, 0.0, 0.0, 0.0, vth);
  const auto view =
      vlasov::StridedDistribution::contiguous(f.data(), p.nx, p.nvx, 0, p.nvy);
  vlasov::ReductionOptions opt;
  opt.comm = MPI_COMM_SELF;
  opt.v_thermal = vth;
  const auto mom = vlasov::reduce_velocity(p, view, opt);

  const double pref = n0 / (2.0 * kPi * vth * vth);
  const double s_exact = -n0 * p.Lx * (std::log(pref) - 1.0);
  INFO("entropy " << mom.entropy << "  exact " << s_exact);
  // Measured relative error 2e-14 on this box; 1e-10 is the quadrature
  // floor of summing 128^2 positive terms, not a wish.
  REQUIRE(mom.entropy == Approx(s_exact).epsilon(1e-10));
  REQUIRE(mom.number == Approx(n0 * p.Lx).epsilon(1e-12));
}

TEST_CASE("the electrostatic reduction holds Ey and Bz at zero",
          "[unit][maxwell][electrostatic]") {
  // The Vlasov-Poisson path is a runtime reduction of the same stepper:
  // update_fields must zero the transverse pair rather than leave whatever
  // was in them. Poisoning Ey, Bz and then taking one field update is the
  // whole test; a no-op reduction would keep the poison.
  if (world_size() != 1) {
    SKIP("single-rank electrostatic reduction check");
  }
  SimParams p;
  p.nx = 16;
  p.nvx = 16;
  p.nvy = 16;
  p.Lx = 2.0 * kPi;
  p.v_max = 0.4;
  p.electrostatic = true;
  p.interp_order = 3;
  p.v_thermal = 0.05;
  vlasov::PhaseSpace ps(p, 2, MPI_COMM_SELF);
  ps.initialise(0, [&](double x, double vx, double vy) {
    return vlasov::ics::maxwellian(vx, vy, 0.05, 1.0 + 0.02 * std::cos(x));
  });
  vlasov::Stepper st(p, ps);
  for (double &v : st.fields.Ey) v = 0.3;
  for (double &v : st.fields.Bz) v = -0.2;
  st.deposit_all();
  st.update_fields(0.01);
  for (int i = 0; i < p.nx; ++i) {
    REQUIRE(st.fields.Ey[static_cast<std::size_t>(i)] == 0.0);
    REQUIRE(st.fields.Bz[static_cast<std::size_t>(i)] == 0.0);
  }
  // And it has to stay zero under a full Strang step, not only the field
  // half: advect_vy with a leftover Bz would rotate the plasma.
  st.advance(0.01);
  for (int i = 0; i < p.nx; ++i) {
    REQUIRE(st.fields.Ey[static_cast<std::size_t>(i)] == 0.0);
    REQUIRE(st.fields.Bz[static_cast<std::size_t>(i)] == 0.0);
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
