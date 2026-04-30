// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file euler.hpp
 * @brief Explicit forward-Euler stepper for the point-wise abstraction.
 *
 * @details
 * `EulerStepper` wraps the canonical point-wise loop
 *
 *     du[{i,j,k}] = model.rhs(t, eval(i,j,k))
 *     u           += dt * du
 *
 * into a single explicit-Euler step. It is decoupled from the physics
 * (`Model`) and the spatial discretization (any `Eval` accepted by
 * `pfc::sim::for_each_interior`). Further methods (RK2, RK4, IMEX) belong in
 * sibling files in this folder under `pfc::sim::steppers::`.
 *
 * The stepper does **not** perform halo exchange or any other backend
 * pre-processing: that is the application's responsibility (FD needs a halo
 * exchange before each step; spectral does not). This keeps the stepper
 * agnostic of FD vs. spectral.
 *
 * Cells outside the interior of `Eval` are left at their previous value
 * because their `du` slot stays at 0 between steps (the internally-managed
 * `du` buffer is value-initialized once at construction; subsequent steps
 * only overwrite the interior).
 *
 * @see openpfc/kernel/simulation/for_each_interior.hpp for the driver loop
 * @see openpfc/kernel/field/grad_point.hpp for the per-point interface
 */

#include <cstddef>
#include <vector>

#include <openpfc/kernel/simulation/for_each_interior.hpp>

namespace pfc::sim::steppers {

template <class Eval, class Model> class EulerStepper {
public:
  EulerStepper(Eval &eval, const Model &model, double dt, std::size_t local_size)
      : m_eval(&eval), m_model(&model), m_dt(dt), m_du(local_size, 0.0) {}

  /** Advance `u` by one explicit-Euler step in place; returns the new time. */
  double step(double t, std::vector<double> &u) {
    pfc::sim::for_each_interior(*m_model, *m_eval, m_du.data(), t);
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(u.size());
    for (std::ptrdiff_t li = 0; li < n; ++li) {
      u[static_cast<std::size_t>(li)] += m_dt * m_du[static_cast<std::size_t>(li)];
    }
    return t + m_dt;
  }

  double dt() const noexcept { return m_dt; }

private:
  Eval *m_eval{nullptr};
  const Model *m_model{nullptr};
  double m_dt{0.0};
  std::vector<double> m_du;
};

/** Class-template argument deduction helper. */
template <class Eval, class Model>
EulerStepper(Eval &, const Model &, double, std::size_t)
    -> EulerStepper<Eval, Model>;

} // namespace pfc::sim::steppers
