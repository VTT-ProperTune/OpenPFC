// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file heat_model.hpp
 * @brief Self-contained heat-equation model.
 *
 * `HeatModel` is the **one place** a physicist needs to look at to understand
 * the problem this app solves: it carries the physical parameters (just `D`
 * here), the initial condition as a runtime-swappable spatial lambda, an
 * optional boundary-value provider for future Dirichlet/Neumann support, and
 * the per-point right-hand side as a direct method `rhs(t, g)`.
 *
 * Design notes:
 *  - `rhs` is a regular method, not `operator()` and not `std::function`, so
 *    the inner `for_each_interior` loop inlines it as cleanly as a free
 *    function. This is the per-cell hot path.
 *  - `initial_condition` is `std::function` because it is set once at startup
 *    (or swapped between named ICs at runtime) and applied per inbox cell at
 *    most once per simulation.
 *  - `boundary_value` is reserved for a future framework-side BC application
 *    helper. With it left empty, the discretization stays effectively
 *    periodic (as today's `heat3d`).
 *  - The struct is intentionally trivial to unit-test without any
 *    discretization, MPI, or FFT context.
 */

#include <cmath>
#include <functional>

#include <heat3d/discretization.hpp>

namespace heat3d {

/** Spatial lambda \f$f(x,y,z)\f$. */
using PointFn = std::function<double(double, double, double)>;
/** Space-time lambda \f$f(x,y,z,t)\f$ (for boundary-value providers). */
using PointFnT = std::function<double(double, double, double, double)>;

/**
 * @brief Heat equation \f$\partial_t u = D \nabla^2 u\f$, self-contained.
 */
struct HeatModel {
  /** Diffusion coefficient. */
  double D = 1.0;

  /**
   * @brief Initial condition \f$u(x,y,z,0)\f$.
   *
   * Default is the Gaussian \f$\exp(-(x^2+y^2+z^2)/(4D))\f$ matching the
   * existing `heat3d` reference IC; assign a different lambda to swap it.
   */
  PointFn initial_condition = [](double x, double y, double z) {
    return std::exp(-(x * x + y * y + z * z) / 4.0);
  };

  /**
   * @brief Optional Dirichlet/Neumann boundary value \f$u_b(x,y,z,t)\f$.
   *
   * Empty by default — the discretization treats the domain as periodic
   * (FD freezes its halo region, spectral assumes periodicity). When set,
   * a future framework helper would impose this value on the appropriate
   * boundary cells each step.
   */
  PointFnT boundary_value{};

  /**
   * @brief Per-point right-hand side \f$\partial_t u = D\nabla^2 u\f$.
   *
   * Hot path: called once per interior cell per time step. `inline` and
   * `noexcept` so the surrounding `for_each_interior` loop fuses into a
   * single tight kernel.
   */
  inline double rhs(double /*t*/, const GradPoint &g) const noexcept {
    return D * (g.uxx + g.uyy + g.uzz);
  }
};

} // namespace heat3d
