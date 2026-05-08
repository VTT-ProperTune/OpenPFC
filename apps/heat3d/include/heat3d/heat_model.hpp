// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file heat_model.hpp
 * @brief 3D heat equation \f$\partial_t u = D \nabla^2 u\f$ — physics model.
 *
 * @details
 * This is the **only** file a physicist needs to edit to define a heat
 * problem on top of OpenPFC. It is intentionally *self-consistent*:
 * the per-point grads aggregate (`HeatGrads`), the diffusion coefficient
 * (`kD`), the initial condition, the optional boundary-value provider,
 * and the per-cell RHS all live next to each other in one short file.
 *
 *  - `HeatGrads` is the per-point grads aggregate the kernel materialises
 *    for the model. The OpenPFC evaluators
 *    (`pfc::field::FdGradient<G>`, `pfc::field::SpectralGradient<G>`) are
 *    templated on `G` and use the `pfc::field::has_*` concepts to fill
 *    only the members `G` declares — naming `xx, yy, zz` here is enough
 *    to get exactly those second derivatives and nothing else.
 *  - `kD` is a single source-level `inline constexpr double` at namespace
 *    scope, not a mutable model member. The five heat3d binaries
 *    (`heat3d_fd`, `heat3d_fd_manual`, `heat3d_fd_scratch`,
 *    `heat3d_spectral`, `heat3d_spectral_pointwise`) share one fixed
 *    value of \f$D\f$ so their L2-vs-analytic outputs are directly
 *    comparable. To experiment with a different coefficient, change the
 *    literal here and rebuild.
 *  - `HeatModel::initial_condition` is a runtime-swappable spatial
 *    lambda (`PointFn = (x, y, z) -> u`).
 *  - `HeatModel::boundary_value` is a space-time provider
 *    (`PointFnT = (x, y, z, t) -> u`) reserved for future
 *    Dirichlet/Neumann support; default-constructed empty.
 *  - `HeatModel::rhs(t, g)` is a plain `inline noexcept` member (not
 *    `operator()` and not `std::function`) so the inner
 *    `pfc::sim::for_each_interior` loop in the application driver inlines
 *    it as cleanly as a free function would.
 *
 * The struct is intentionally header-only and **free of any
 * `<openpfc/...>` include**: only `<cmath>` and `<functional>` from the
 * C++ standard library are needed. This makes the model trivial to
 * unit-test in isolation (see `apps/heat3d/tests/test_heat3d.cpp`) and
 * reusable from any OpenPFC application driver.
 *
 * @see openpfc/kernel/field/grad_concepts.hpp for the per-member detection
 *      concepts the kernel uses to introspect `HeatGrads`.
 * @see openpfc/kernel/field/grad_point.hpp for the convenience default
 *      catalog struct apps can use instead when minimising kernel work
 *      isn't critical.
 */

#include <cmath>
#include <functional>

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
 * Change the literal here (and rebuild) to experiment with a different
 * coefficient — every driver, the analytic reference solution, and the
 * default initial condition pick it up automatically.
 */
inline constexpr double kD = 1.0;

/**
 * @brief Minimal per-point grads aggregate for the heat equation.
 *
 * The heat equation \f$\partial_t u = D\nabla^2 u\f$ needs only the three
 * unmixed second derivatives of `u`, so this aggregate names exactly
 * those slots from the catalog
 * `{ value, x, y, z, xx, yy, zz, xy, xz, yz }` recognised by the OpenPFC
 * kernel. Default-initialised to zero so any evaluator that fails to
 * populate a slot (today: none — both FD and spectral fill every member
 * declared here) still yields a well-defined RHS evaluation.
 */
struct HeatGrads {
  double xx{};
  double yy{};
  double zz{};
};

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
