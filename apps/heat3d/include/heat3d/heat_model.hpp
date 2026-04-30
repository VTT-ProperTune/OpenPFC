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
 *  - the physical parameter (just the diffusion coefficient `kD`),
 *  - the initial condition as a runtime-swappable spatial lambda
 *    (`PointFn = (x, y, z) -> u`),
 *  - an optional boundary-value provider for future Dirichlet/Neumann
 *    support (`PointFnT = (x, y, z, t) -> u`),
 *  - the per-point right-hand side as a regular method `rhs(t, g)`.
 *
 * `rhs` is a plain `inline noexcept` member (not `operator()` and not
 * `std::function`) so the inner `pfc::sim::for_each_interior` loop in the
 * application driver inlines it as cleanly as a free function would.
 *
 * The diffusion coefficient is a **single source-level `inline constexpr
 * double kD`** at namespace scope, not a mutable member: the heat3d
 * binaries (`heat3d_fd`, `heat3d_fd_manual`, `heat3d_fd_scratch`,
 * `heat3d_spectral`, `heat3d_spectral_pointwise`) are educational
 * examples sharing one fixed value of \f$D\f$ so their L2-vs-analytic
 * outputs are directly comparable. To experiment with a different
 * coefficient, change the literal here.
 *
 * The struct is intentionally header-only and **free of any
 * `<openpfc/...>` include**: only `<cmath>` and `<functional>` from the
 * C++ standard library are needed. The per-point grads aggregate
 * `heat3d::HeatGrads` is also model-owned (in `heat_grads.hpp`); the
 * OpenPFC kernel introspects that struct via concepts and produces
 * exactly the partial derivatives it names. This makes the model
 * trivial to unit-test in isolation (see
 * `apps/heat3d/tests/test_heat3d.cpp`) and reusable from any OpenPFC
 * application driver.
 */

#include <cmath>
#include <functional>

#include <heat3d/heat_grads.hpp>

namespace heat3d {

/**
 * @brief Spatial coordinate function \f$f(x,y,z)\f$.
 *
 * Type-erased callable used by the IC. Local to `heat3d` so the model
 * file does not depend on OpenPFC.
 */
using PointFn = std::function<double(double, double, double)>;

/**
 * @brief Space-time coordinate function \f$f(x,y,z,t)\f$.
 *
 * Type-erased callable used by the boundary-value provider. Local to
 * `heat3d` so the model file does not depend on OpenPFC.
 */
using PointFnT = std::function<double(double, double, double, double)>;

/**
 * @brief Diffusion coefficient \f$D\f$ shared by every heat3d binary.
 *
 * Hard-pinned to `1.0` to match `examples/15_finite_difference_heat.cpp`
 * and to keep the L2 numbers reported by the five drivers comparable.
 * Change the literal here (and re-build) to experiment with a different
 * coefficient — every driver, the analytic reference solution, and the
 * default initial condition pick it up automatically.
 */
inline constexpr double kD = 1.0;

/**
 * @brief Heat equation \f$\partial_t u = D \nabla^2 u\f$, self-contained.
 *
 * The diffusion coefficient lives outside the struct (see `heat3d::kD`)
 * so the model has no mutable physical parameters; only the initial
 * condition and the boundary-value provider are user-tunable.
 */
struct HeatModel {
  /**
   * @brief Initial condition \f$u(x,y,z,0)\f$.
   *
   * Default: the fundamental Gaussian solution at \f$t=0\f$ for the
   * configured diffusion coefficient,
   * \f$u_0(\mathbf{x}) = \exp\!\bigl(-|\mathbf{x}|^2/(4D)\bigr)\f$.
   *
   * The lambda reads `kD` directly (no `this` capture), so the model is
   * trivially copyable / movable.
   */
  PointFn initial_condition = [](double x, double y, double z) {
    return std::exp(-(x * x + y * y + z * z) / (4.0 * kD));
  };

  /**
   * @brief Optional Dirichlet/Neumann boundary value \f$u_b(x,y,z,t)\f$.
   *
   * Empty by default — the discretization treats the domain as periodic
   * (FD freezes its halo region, spectral assumes periodicity).
   */
  PointFnT boundary_value{};

  /** Per-point right-hand side \f$\partial_t u = D\nabla^2 u\f$ (hot path). */
  [[nodiscard]] double rhs(double /*t*/, const HeatGrads &g) const noexcept {
    return kD * (g.xx + g.yy + g.zz);
  }
};

} // namespace heat3d
