// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file euler.hpp
 * @brief Explicit forward-Euler stepper for arbitrary point-wise RHS callables.
 *
 * @details
 * `EulerStepper` is a **pure ODE integrator** that applies one forward-Euler
 * step in place,
 *
 *     u += dt * rhs(t, u)
 *
 * It owns nothing more than `dt`, an internal scratch `du` buffer, and a
 * user-supplied `Rhs` callable. The callable is the only thing that knows
 * about the spatial discretization (FD, spectral, custom, …); the stepper
 * itself is agnostic.
 *
 * `Rhs` must be invocable as
 *
 *     rhs(double t, std::vector<double>& u, std::vector<double>& du)
 *
 * and is expected to **fill** `du` (sized `local_size` by the constructor).
 * `u` is passed read-only by convention; the stepper performs the
 * `u += dt * du` accumulation itself. Cells that `rhs` leaves untouched keep
 * their previous `du` value (the buffer is value-initialized once at
 * construction; subsequent steps overwrite whatever the RHS chooses to
 * overwrite). The stepper does not perform halo exchange or any other
 * backend pre-processing — that is the application's responsibility (FD
 * needs a halo exchange before each step; spectral does not).
 *
 * Most applications do not construct `EulerStepper` directly. Use one of
 * the `pfc::sim::steppers::create` factories at the bottom of this file to
 * bind a model + gradient evaluator to the canonical
 * `for_each_interior(model, eval, du, t)` RHS. They mirror the
 * `world::create`, `decomposition::create`, `fft::create`, `field::create`
 * convention used throughout OpenPFC.
 *
 * Further methods (RK2, RK4, IMEX) belong in sibling files in this folder
 * under `pfc::sim::steppers::`.
 *
 * @see openpfc/kernel/simulation/for_each_interior.hpp for the canonical
 *      point-wise driver loop the `create` factories wrap
 * @see openpfc/kernel/field/grad_point.hpp for the per-point interface
 * @see openpfc/kernel/field/local_field.hpp for the typed field bundle
 *      that the `LocalField` overload derives `local_size` from
 */

#include <cstddef>
#include <utility>
#include <vector>

#include <openpfc/kernel/field/local_field.hpp>
#include <openpfc/kernel/simulation/for_each_interior.hpp>

namespace pfc::sim::steppers {

/**
 * @brief Pure forward-Euler ODE stepper: `u += dt * rhs(t, u)`.
 *
 * @tparam Rhs Any callable invocable as
 *             `rhs(double t, std::vector<double>& u, std::vector<double>& du)`.
 *             It must fill `du`; the stepper adds `dt * du` to `u`.
 */
template <class Rhs> class EulerStepper {
public:
  EulerStepper(double dt, std::size_t local_size, Rhs rhs)
      : m_dt(dt), m_du(local_size, 0.0), m_rhs(std::move(rhs)) {}

  /** Advance `u` by one explicit-Euler step in place; returns the new time. */
  double step(double t, std::vector<double> &u) {
    m_rhs(t, u, m_du);
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(u.size());
    for (std::ptrdiff_t li = 0; li < n; ++li) {
      u[static_cast<std::size_t>(li)] += m_dt * m_du[static_cast<std::size_t>(li)];
    }
    return t + m_dt;
  }

  double dt() const noexcept { return m_dt; }

private:
  double m_dt{0.0};
  std::vector<double> m_du;
  Rhs m_rhs;
};

// -----------------------------------------------------------------------------
// `create` free-function factories.
//
// They build an `EulerStepper` whose RHS is the canonical point-wise loop
//
//     du[{i,j,k}] = model.rhs(t, eval(i,j,k))
//
// over the interior cells exposed by `eval`. The stepper itself remains
// agnostic of the (Eval, Model) types — the wiring lives entirely inside the
// captured lambda below.
// -----------------------------------------------------------------------------

/**
 * @brief Build an `EulerStepper` for the canonical point-wise RHS, given the
 *        local buffer size explicitly.
 *
 * Prefer the `LocalField` overload when you have one — it derives
 * `local_size` from `u.size()`.
 *
 * @param eval        Per-point gradient evaluator (e.g. `pfc::field::FdGradient`,
 *                    `pfc::field::SpectralGradient`). Captured by reference;
 *                    must outlive the returned stepper.
 * @param model       Physics model with a method
 *                    `rhs(double t, const pfc::field::GradPoint&) -> double`.
 *                    Captured by reference; must outlive the returned stepper.
 * @param dt          Time-step size.
 * @param local_size  Number of cells in the rank-local field buffer
 *                    (typically `u.size()`).
 */
template <class Eval, class Model>
auto create(Eval &eval, const Model &model, double dt, std::size_t local_size) {
  auto rhs = [&eval, &model](double t, const std::vector<double> & /*u*/,
                             std::vector<double> &du) {
    pfc::sim::for_each_interior(model, eval, du.data(), t);
  };
  return EulerStepper<decltype(rhs)>(dt, local_size, std::move(rhs));
}

/**
 * @brief Build an `EulerStepper` for the canonical point-wise RHS, deriving
 *        the local buffer size from the field bundle.
 *
 * Mirrors the `world::create`, `decomposition::create`, `fft::create`,
 * `field::create` family used elsewhere in OpenPFC.
 *
 * @param u      Local field whose `size()` defines the internal `du` buffer
 *               (and which the application owns). Not stored by the stepper.
 * @param eval   Per-point gradient evaluator. Captured by reference.
 * @param model  Physics model. Captured by reference.
 * @param dt     Time-step size.
 */
template <class T, class Eval, class Model>
auto create(const pfc::field::LocalField<T> &u, Eval &eval, const Model &model,
            double dt) {
  return create(eval, model, dt, u.size());
}

} // namespace pfc::sim::steppers
