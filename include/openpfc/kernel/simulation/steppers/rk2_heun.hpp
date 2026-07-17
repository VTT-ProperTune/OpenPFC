// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file rk2_heun.hpp
 * @brief Explicit RK2 (Heun's method) stepper for arbitrary point-wise RHS
 * callables.
 *
 * @details
 * `RK2HeunStepper` is a **pure ODE integrator** that applies one RK2 Heun step
 * in place,
 *
 *     predictor: u_p = u + dt * rhs(t, u)
 *     corrector: u += dt/2 * (rhs(t, u) + rhs(t + dt, u_p))
 *
 * It owns nothing more than `dt`, internal scratch buffers (`m_du`, `m_predictor`,
 * `m_rhs_predictor`), and a user-supplied `Rhs` callable. The callable is the only
 * thing that knows about the spatial discretization (FD, spectral, custom, ...);
 * the stepper itself is agnostic.
 *
 * `Rhs` must be invocable as
 *
 *     rhs(double t, std::vector<double>& u, std::vector<double>& du)
 *
 * and is expected to **fill** `du` (sized `local_size` by the constructor).
 * `u` is passed read-only by convention; the stepper performs the accumulation
 * itself. Cells that `rhs` leaves untouched keep their previous `du` value (the
 * buffer is value-initialized once at construction; subsequent steps overwrite
 * whatever the RHS chooses to overwrite). The stepper does not perform halo
 * exchange or any other backend pre-processing — that is the application's
 * responsibility (FD needs a halo exchange before each step; spectral does not).
 *
 * The stepper pre-allocates two RHS buffers (`m_du`, `m_rhs_predictor`) to avoid
 * per-step allocation, mirroring `EulerStepper`'s single `m_du` buffer design.
 * The corrector step reuses `m_du` from the predictor step instead of recomputing
 * `rhs(t, u)`.
 *
 * ## Algorithm
 *
 * Given the ODE `du/dt = f(t, u)`, Heun's method computes:
 *
 * 1. **Predictor step**: `u_p = u + dt * f(t, u)`
 * 2. **Corrector step**: `u = u + dt/2 * (f(t, u) + f(t + dt, u_p))`
 *
 * This is a second-order Runge-Kutta method with improved accuracy compared to
 * forward-Euler, at the cost of two RHS evaluations per step.
 *
 * ## Usage Pattern
 *
 * ```cpp
 * using namespace pfc::sim::steppers;
 *
 * // RHS: du/dt = some function of t and u
 * auto rhs = [](double t, const std::vector<double>& u, std::vector<double>& du) {
 *     for (std::size_t i = 0; i < u.size(); ++i) {
 *         du[i] = -u[i];  // Simple decay
 *     }
 * };
 *
 * RK2HeunStepper stepper(dt, local_size, rhs);
 *
 * while (!done) {
 *     t = stepper.step(t, u);
 *     // ... process u at new time t ...
 * }
 * ```
 *
 * @see euler.hpp for the forward-Euler stepper
 * @see openpfc/kernel/simulation/for_each_interior.hpp for the canonical
 *      point-wise driver loop
 */

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include <openpfc/kernel/simulation/steppers/stage_protocol.hpp>

namespace pfc::sim::steppers {

/**
 * @brief Explicit RK2 (Heun's method) stepper: `u += dt/2 * (rhs(t, u) + rhs(t + dt,
 * u_p))`.
 *
 * @tparam Rhs Any callable invocable as
 *             `rhs(double t, std::vector<double>& u, std::vector<double>& du)`.
 *             It must fill `du`; the stepper performs the RK2 accumulation.
 *
 * ## Algorithm
 *
 * 1. **Predictor**: `u_p = u + dt * rhs(t, u)`
 * 2. **Corrector**: `u += dt/2 * (rhs(t, u) + rhs(t + dt, u_p))`
 *
 * The corrector step reuses `rhs(t, u)` from the predictor (stored in `m_du`),
 * requiring only one additional RHS evaluation for `rhs(t + dt, u_p)`.
 *
 * ## Memory Layout
 *
 * Pre-allocates three buffers:
 * - `m_du`: RHS evaluation at `(t, u)` (reused in corrector)
 * - `m_predictor`: Predictor state `u_p`
 * - `m_rhs_predictor`: RHS evaluation at `(t + dt, u_p)`
 *
 * This design avoids per-step allocations and mirrors `EulerStepper`'s
 * pre-allocation strategy.
 */
template <class Rhs>
  requires StageFunction<Rhs>
class RK2HeunStepper {
public:
  /**
   * @brief Construct an RK2 Heun stepper.
   *
   * @param dt Time step size (must be positive)
   * @param local_size Number of elements in the local state vector `u`
   * @param rhs Right-hand side callable, invocable as
   *            `rhs(double t, const std::vector<double>& u, std::vector<double>&
   * du)`
   *
   * @post All buffers are value-initialized to 0.0
   */
  RK2HeunStepper(double dt, std::size_t local_size, Rhs rhs)
      : m_dt(dt), m_du(local_size, 0.0), m_predictor(local_size, 0.0),
        m_rhs_predictor(local_size, 0.0), m_rhs(std::move(rhs)) {}

  /**
   * @brief Advance `u` by one RK2 Heun step in place; returns the new time.
   *
   * ## Algorithm
   *
   * 1. **Predictor**: Evaluate RHS at `(t, u)` and compute predictor state
   *    ```
   *    rhs(t, u, m_du)
   *    m_predictor = u + dt * m_du
   *    ```
   * 2. **Corrector**: Evaluate RHS at `(t + dt, m_predictor)` and apply
   *    trapezoidal update
   *    ```
   *    rhs(t + dt, m_predictor, m_rhs_predictor)
   *    u += dt/2 * (m_du + m_rhs_predictor)
   *    ```
   *
   * @param t Current time
   * @param u State vector (modified in place)
   * @return New time `t + dt`
   *
   * @pre `u.size()` must equal the `local_size` passed to the constructor
   * @post `u` contains the advanced state at time `t + dt`
   */
  double step(double t, std::vector<double> &u) {
    // Predictor step: u_p = u + dt * rhs(t, u)
    m_rhs(t, u, m_du);
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(u.size());
    for (std::ptrdiff_t li = 0; li < n; ++li) {
      m_predictor[static_cast<std::size_t>(li)] =
          u[static_cast<std::size_t>(li)] +
          m_dt * m_du[static_cast<std::size_t>(li)];
    }

    // Corrector step: u += dt/2 * (rhs(t, u) + rhs(t + dt, u_p))
    // m_du already contains rhs(t, u) from predictor - reuse it
    m_rhs(t + m_dt, m_predictor, m_rhs_predictor);
    for (std::ptrdiff_t li = 0; li < n; ++li) {
      u[static_cast<std::size_t>(li)] +=
          0.5 * m_dt *
          (m_du[static_cast<std::size_t>(li)] +
           m_rhs_predictor[static_cast<std::size_t>(li)]);
    }

    return t + m_dt;
  }

  /**
   * @brief Get the time step size.
   *
   * @return Time step `dt`
   */
  double dt() const noexcept { return m_dt; }

private:
  double m_dt{0.0};                    ///< Time step size
  std::vector<double> m_du;            ///< RHS at (t, u) - reused in corrector
  std::vector<double> m_predictor;     ///< Predictor state u_p
  std::vector<double> m_rhs_predictor; ///< RHS at (t + dt, u_p)
  Rhs m_rhs;                           ///< Right-hand side callable
};

} // namespace pfc::sim::steppers
