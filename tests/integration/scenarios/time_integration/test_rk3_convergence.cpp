// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_rk3_convergence.cpp
 * @brief Temporal-convergence integration test for `RK3HeunStepper`.
 *
 * @details
 * This test measures the *observed order of convergence* of
 * `pfc::sim::steppers::RK3HeunStepper` against a known analytical solution
 * of the 1D heat equation `du/dt = D * d2u/dx2` on a periodic domain:
 *
 *     u(x, t) = exp(-k^2 * D * t) * sin(k * x)
 *
 * for a single spatial Fourier mode `k`. This is an *exact* solution of the
 * continuous PDE for any `k`, `D`, and `t` — a standard benchmark for
 * verifying temporal integrators (see e.g. any numerical PDE textbook's
 * treatment of the heat equation via separation of variables).
 *
 * ## Why the spatial operator here is exact, not a finite-difference stencil
 *
 * The naive way to build this benchmark is to discretize `d2u/dx2` with a
 * standard second-order central finite difference,
 * `(u[i+1] - 2u[i] + u[i-1]) / dx^2`, and let `RK3HeunStepper` integrate the
 * resulting semi-discrete ODE system in time. That construction was tried
 * repeatedly and always failed the convergence assertion below — not
 * because the stepper's math was wrong, but because of an unavoidable
 * tension between *explicit stability* and *spatial truncation error* for
 * this particular stencil, at the dt values this test is required to use:
 *
 * - The centered 2nd-order Laplacian stencil's worst-case (Nyquist-mode)
 *   eigenvalue scales as `-4D/dx^2`. Explicit RK3 requires
 *   `dt * 4D/dx^2` to stay inside its stability region (real-axis limit
 *   ~ -2.51), i.e. the *mesh Fourier number* `R = dt*D/dx^2` is bounded by a
 *   fixed constant, independent of how the mesh is chosen.
 * - The stencil's *relative eigenvalue error* for the sampled mode `k` is
 *   `~ (k*dx)^2 / 12` (the usual 2nd-order truncation error), while the
 *   *temporal* error signal this test wants to measure scales as
 *   `z0 = D*k^2*dt = R * (k*dx)^2`.
 * - Dividing the two shows `z0 / spatial_error ~ 12*R`, which is capped by
 *   the *same* stability bound on `R` — so, for a 2nd-order central FD
 *   Laplacian, the temporal error this test wants to isolate can *never* be
 *   made much larger than the (dt-independent) spatial truncation-error
 *   floor at these dt values, no matter how the grid is chosen: making the
 *   grid coarser keeps the spatial floor large and flat (observed order
 *   ~ 0, as the measured error asymptotes to that floor exactly as
 *   warned against), while making it finer at fixed dt violates the
 *   stability bound and blows up outright (floating-point round-off seeds
 *   the unstable Nyquist mode, which is amplified by a factor of roughly
 *   `|dt*4D/dx^2|^3/6` per step once outside the stability region).
 *
 * The fix is to remove the spatial truncation error from the picture
 * entirely rather than merely trying to shrink it: since the state stays
 * exactly proportional to `sin(k*x)` for all time (the true PDE, and any
 * *linear* semi-discretization that treats `sin(k*x)` as an eigenfunction,
 * preserves this 1-D eigenspace), the spatial operator can be evaluated by
 * its *exact*, non-discretized action on that eigenspace,
 * `d2/dx2 [c * sin(k*x)] = -k^2 * [c * sin(k*x)]`, i.e. `du[i] = -D*k^2*u[i]`
 * pointwise. This is mathematically the spectral derivative of a fully
 * resolved single-mode signal (zero truncation error to machine precision)
 * and carries no dx-dependent stability constraint at all (each grid point
 * evolves as an independent copy of the same scalar ODE, so there is no
 * neighbor-coupled high-wavenumber mode for round-off to seed). What
 * remains is *purely* the RK3 stepper's own `O(dt^3)` global truncation
 * error — exactly what this test is supposed to measure.
 *
 * A grid of `N` points is still used, to build the initial condition and
 * the reference solution and to compute a genuine (root-mean-square) L2
 * error over many spatial samples rather than a single point.
 */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

#include <openpfc/kernel/simulation/steppers/rk3_heun.hpp>

using namespace pfc::sim::steppers;

namespace {

constexpr double kD = 1.0;          // Diffusion coefficient
constexpr double kK = 2.0 * M_PI;   // Wavenumber (one period over L=1)
constexpr double kL = 1.0;          // Domain length
constexpr std::size_t kN = 128;     // Spatial samples for the L2 error
constexpr double kFinalTime = 0.01; // T

// Exact solution of du/dt = D * d2u/dx2 on x in [0, L), periodic BC.
double exact_solution(double x, double t) {
  return std::exp(-kK * kK * kD * t) * std::sin(kK * x);
}

// Run RK3HeunStepper from t=0 to kFinalTime with the given dt and return the
// root-mean-square (L2) error against the exact solution at the sampled grid
// points.
double run_and_measure_l2_error(double dt) {
  const double dx = kL / static_cast<double>(kN);

  std::vector<double> u(kN);
  for (std::size_t i = 0; i < kN; ++i) {
    const double x = static_cast<double>(i) * dx;
    u[i] = exact_solution(x, 0.0);
  }

  // Exact (non-discretized) spatial operator: since `u` stays exactly
  // proportional to sin(k*x) throughout the integration (see file-level
  // comment above), d2u/dx2 = -k^2 * u pointwise, with zero spatial
  // truncation error. `t` is unused: the operator is time-independent.
  auto rhs = [](double /*t*/, const std::vector<double> &u_in,
                std::vector<double> &du) {
    for (std::size_t i = 0; i < u_in.size(); ++i) {
      du[i] = -kD * kK * kK * u_in[i];
    }
  };

  RK3HeunStepper<decltype(rhs)> stepper(dt, kN, rhs);

  const int n_steps = static_cast<int>(std::lround(kFinalTime / dt));
  double t = 0.0;
  for (int step = 0; step < n_steps; ++step) {
    t = stepper.step(t, u);
  }

  double sum_sq = 0.0;
  for (std::size_t i = 0; i < kN; ++i) {
    const double x = static_cast<double>(i) * dx;
    const double diff = u[i] - exact_solution(x, t);
    sum_sq += diff * diff;
  }
  return std::sqrt(sum_sq / static_cast<double>(kN));
}

} // namespace

TEST_CASE("RK3HeunStepper achieves third-order temporal convergence on 1D "
          "heat equation",
          "[integration][time_integration][rk3_convergence]") {
  const std::vector<double> dts = {0.001, 0.0005, 0.00025};

  std::vector<double> errors;
  errors.reserve(dts.size());
  for (double dt : dts) {
    errors.push_back(run_and_measure_l2_error(dt));
  }

  // Sanity: errors must be finite, non-zero, and shrink monotonically as dt
  // shrinks (a necessary, if not sufficient, condition for convergence).
  for (double err : errors) {
    REQUIRE(std::isfinite(err));
    REQUIRE(err > 0.0);
  }
  REQUIRE(errors[1] < errors[0]);
  REQUIRE(errors[2] < errors[1]);

  // Observed order of convergence between successive dt pairs, via the
  // standard log(e1/e2) / log(dt1/dt2) ratio.
  const double order_1 = std::log(errors[0] / errors[1]) / std::log(dts[0] / dts[1]);
  const double order_2 = std::log(errors[1] / errors[2]) / std::log(dts[1] / dts[2]);
  const double observed_order = 0.5 * (order_1 + order_2);

  INFO("dt = " << dts[0] << ", " << dts[1] << ", " << dts[2]);
  INFO("errors = " << errors[0] << ", " << errors[1] << ", " << errors[2]);
  INFO("order_1 = " << order_1 << ", order_2 = " << order_2);
  INFO("observed_order (average) = " << observed_order);

  REQUIRE(order_1 > 2.8);
  REQUIRE(order_1 < 3.2);
  REQUIRE(order_2 > 2.8);
  REQUIRE(order_2 < 3.2);
  REQUIRE(observed_order > 2.8);
  REQUIRE(observed_order < 3.2);
}
