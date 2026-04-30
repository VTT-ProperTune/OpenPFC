// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file stepper.hpp
 * @brief Time integrators for the point-wise discretization abstraction.
 *
 * A stepper is decoupled from the physics (`Rhs`) and the spatial
 * discretization (any `Eval` accepted by `for_each_interior`). The point-wise
 * pattern
 *
 *     du[{i,j,k}] = rhs(t, eval(i,j,k))
 *     u  += dt * du
 *
 * is a single explicit Euler step. `EulerStepper` packages it; further methods
 * (RK2, RK4, IMEX) belong here as separate types implementing the same shape.
 *
 * The stepper does **not** perform the halo exchange or any other backend
 * pre-processing: that is the application's responsibility (FD needs a halo
 * exchange before the next step; spectral does not). This keeps the stepper
 * agnostic of FD vs. spectral.
 */

#include <cstddef>
#include <vector>

#include <heat3d/discretization.hpp>

namespace heat3d {

/**
 * @brief Explicit forward-Euler stepper for the point-wise abstraction.
 *
 * Holds references to a discretization `Eval`, a callable `Rhs`, and an
 * internally-managed `du` buffer sized to the application's local field. Each
 * call to `step(t, u)` writes `du` for the interior (via `for_each_interior`)
 * then does `u += dt * du` over the full local buffer (cells outside the
 * interior are left at their previous value because their `du` slot stays at 0
 * — this matches the existing `heat3d` behaviour where the per-rank halo
 * region is intentionally not updated).
 *
 * Templated on `Eval` and `Rhs` so the inner loop in `for_each_interior`
 * inlines: no `std::function` indirection.
 */
template <class Eval, class Rhs> class EulerStepper {
public:
  EulerStepper(Eval &eval, Rhs rhs, double dt, std::size_t local_size)
      : m_eval(&eval), m_rhs(rhs), m_dt(dt), m_du(local_size, 0.0) {}

  /** Advance `u` by one explicit-Euler step in place; returns the new time. */
  double step(double t, std::vector<double> &u) {
    for_each_interior(*m_eval, m_du.data(), t, m_rhs);
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(u.size());
    for (std::ptrdiff_t li = 0; li < n; ++li) {
      u[static_cast<std::size_t>(li)] += m_dt * m_du[static_cast<std::size_t>(li)];
    }
    return t + m_dt;
  }

  double dt() const noexcept { return m_dt; }

private:
  Eval *m_eval{nullptr};
  Rhs m_rhs;
  double m_dt{0.0};
  std::vector<double> m_du;
};

/** Class-template argument deduction helper. */
template <class Eval, class Rhs>
EulerStepper(Eval &, Rhs, double, std::size_t) -> EulerStepper<Eval, Rhs>;

} // namespace heat3d
