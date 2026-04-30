// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file heat_model.hpp
 * @brief 3D heat equation \f$\partial_t u = D \nabla^2 u\f$ — physics model.
 *
 * @details
 * This is the **only** file a physicist needs to edit to define a heat
 * problem on top of OpenPFC. It carries:
 *
 *  - the physical parameter (just the diffusion coefficient `D`),
 *  - the initial condition as a runtime-swappable spatial lambda
 *    (`field::PointFn = (x, y, z) -> u`),
 *  - an optional boundary-value provider for future Dirichlet/Neumann
 *    support (`field::PointFnT = (x, y, z, t) -> u`),
 *  - the per-point right-hand side as a regular method `rhs(t, g)`.
 *
 * `rhs` is a plain `inline noexcept` member (not `operator()` and not
 * `std::function`) so the inner `pfc::sim::for_each_interior` loop in the
 * application driver inlines it as cleanly as a free function would.
 *
 * The struct is intentionally header-only and free of MPI / FFT / OpenMP
 * dependencies — that makes it trivial to unit-test in isolation
 * (see `apps/heat3d/tests/test_heat3d.cpp`) and reusable from any
 * OpenPFC application driver.
 */

#include <cmath>

#include <openpfc/kernel/field/grad_point.hpp>
#include <openpfc/kernel/field/operations.hpp>

namespace heat3d {

/**
 * @brief Heat equation \f$\partial_t u = D \nabla^2 u\f$, self-contained.
 */
struct HeatModel {
  /** Diffusion coefficient. */
  double D = 1.0;

  /**
   * @brief Initial condition \f$u(x,y,z,0)\f$.
   *
   * Default: the fundamental Gaussian solution at \f$t=0\f$ for the
   * configured diffusion coefficient,
   * \f$u_0(\mathbf{x}) = \exp\!\bigl(-|\mathbf{x}|^2/(4D)\bigr)\f$.
   *
   * The lambda captures `this` so that updating `model.D` after
   * construction is automatically reflected in the IC the next time it is
   * sampled. (Implication: `HeatModel` instances must not be copied or
   * moved while their `initial_condition` is in use — the captured `this`
   * would still reference the source object.)
   */
  pfc::field::PointFn initial_condition = [this](double x, double y, double z) {
    return std::exp(-(x * x + y * y + z * z) / (4.0 * D));
  };

  /**
   * @brief Optional Dirichlet/Neumann boundary value \f$u_b(x,y,z,t)\f$.
   *
   * Empty by default — the discretization treats the domain as periodic
   * (FD freezes its halo region, spectral assumes periodicity).
   */
  pfc::field::PointFnT boundary_value{};

  /** Per-point right-hand side \f$\partial_t u = D\nabla^2 u\f$ (hot path). */
  inline double rhs(double /*t*/, const pfc::field::GradPoint &g) const noexcept {
    return D * (g.uxx + g.uyy + g.uzz);
  }
};

} // namespace heat3d
