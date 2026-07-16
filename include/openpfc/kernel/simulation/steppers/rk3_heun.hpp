// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file rk3_heun.hpp
 * @brief Explicit RK3 (Heun's third-order method) stepper for arbitrary
 *        point-wise RHS callables.
 *
 * @details
 * `RK3HeunStepper` is a **pure ODE integrator** that applies one step of
 * Heun's third-order explicit Runge-Kutta method in place,
 *
 *     k1 = rhs(t, u)
 *     k2 = rhs(t + dt/3,   u + dt * (1/3) * k1)
 *     k3 = rhs(t + 2*dt/3, u + dt * (2/3) * k2)
 *     u += dt * (1/4) * k1 + dt * (3/4) * k3
 *
 * It owns nothing more than `dt`, internal scratch buffers (`m_k1`, `m_k2`,
 * `m_u_temp`), and a user-supplied `Rhs` callable. The callable is the only
 * thing that knows about the spatial discretization (FD, spectral, custom,
 * ...); the stepper itself is agnostic.
 *
 * `Rhs` must be invocable as
 *
 *     rhs(double t, const std::vector<double>& u, std::vector<double>& du)
 *
 * and is expected to **fill** `du` (sized `local_size` by the constructor).
 * `u` is passed read-only by convention; the stepper performs the
 * accumulation itself. Cells that `rhs` leaves untouched keep their previous
 * `du` value (the buffer is value-initialized once at construction;
 * subsequent steps overwrite whatever the RHS chooses to overwrite). The
 * stepper does not perform halo exchange or any other backend
 * pre-processing — that is the application's responsibility (FD needs a
 * halo exchange before each stage; spectral does not).
 *
 * ## Algorithm
 *
 * Given the ODE `du/dt = f(t, u)`, Heun's third-order method computes three
 * stage derivatives at `c = (0, 1/3, 2/3)` and combines them with weights
 * `b = (1/4, 0, 3/4)`:
 *
 * 1. **Stage 1**: `k1 = f(t, u)`
 * 2. **Stage 2**: `k2 = f(t + dt/3, u + dt/3 * k1)`
 * 3. **Stage 3**: `k3 = f(t + 2*dt/3, u + 2*dt/3 * k2)` — note this stage
 *    depends **only** on `k2`, not `k1`; there is no `k1` contribution to
 *    the stage-3 evaluation point.
 * 4. **Combination**: `u = u + dt/4 * k1 + dt*3/4 * k3` — note `k2` does
 *    **not** appear in the final combination; it is used only to evaluate
 *    the stage-3 point.
 *
 * This is a third-order Runge-Kutta method: the local truncation error is
 * `O(dt^4)` and the global error is `O(dt^3)`, at the cost of three RHS
 * evaluations per step (versus two for `RK2HeunStepper`).
 *
 * ## Memory Layout
 *
 * Pre-allocates exactly three buffers:
 * - `m_k1`: RHS evaluation at `(t, u)`, kept live for the whole step because
 *   the final combination needs it.
 * - `m_k2`: RHS evaluation at `(t + dt/3, u + dt/3*k1)`. Once the stage-3
 *   evaluation point has been built from it, `k2`'s value is never read
 *   again (it does not participate in the final combination), so this same
 *   buffer is **reused in place** to hold `k3 = f(t + 2*dt/3, ...)`. This
 *   keeps the stepper to exactly three buffers instead of four, without
 *   aliasing: by the time `m_k2` is overwritten with `k3`, nothing else
 *   still needs the old `k2` value.
 * - `m_u_temp`: Staging buffer for the stage-2 and stage-3 evaluation
 *   points (`u + dt/3*k1`, then `u + 2*dt/3*k2`). Both are computed from
 *   the *original* `u` (never mutated mid-step), so overwriting
 *   `m_u_temp` in place between stages is safe.
 *
 * This design avoids per-step allocation and mirrors `RK2HeunStepper`'s
 * pre-allocation strategy, while exploiting the fact that Heun's
 * third-order tableau has a zero weight on `k2` in the final combination to
 * avoid needing a fourth buffer.
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
 * RK3HeunStepper stepper(dt, local_size, rhs);
 *
 * while (!done) {
 *     t = stepper.step(t, u);
 *     // ... process u at new time t ...
 * }
 * ```
 *
 * @see rk2_heun.hpp for the second-order Heun stepper
 * @see euler.hpp for the forward-Euler stepper
 * @see openpfc/kernel/simulation/for_each_interior.hpp for the canonical
 *      point-wise driver loop
 */

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace pfc::sim::steppers {

/**
 * @brief Explicit RK3 (Heun's third-order method) stepper:
 *        `u += dt/4 * k1 + dt*3/4 * k3`.
 *
 * @tparam Rhs Any callable invocable as
 *             `rhs(double t, std::vector<double>& u, std::vector<double>& du)`.
 *             It must fill `du`; the stepper performs the RK3 accumulation.
 *
 * ## Algorithm
 *
 * 1. **Stage 1**: `k1 = rhs(t, u)`
 * 2. **Stage 2**: `k2 = rhs(t + dt/3, u + dt/3 * k1)`
 * 3. **Stage 3**: `k3 = rhs(t + 2*dt/3, u + 2*dt/3 * k2)` (no `k1` term)
 * 4. **Combination**: `u += dt/4 * k1 + dt*3/4 * k3` (no `k2` term)
 *
 * ## Memory Layout
 *
 * Pre-allocates three buffers:
 * - `m_k1`: RHS at `(t, u)` (kept live for the final combination)
 * - `m_k2`: RHS at `(t + dt/3, u + dt/3*k1)`, later reused in place to hold
 *   RHS at `(t + 2*dt/3, u + 2*dt/3*k2)` once the old `k2` value is no
 *   longer needed
 * - `m_u_temp`: Staging buffer for the stage-2 and stage-3 evaluation points
 */
template <class Rhs> class RK3HeunStepper {
public:
  /**
   * @brief Construct an RK3 Heun stepper.
   *
   * @param dt Time step size (must be positive)
   * @param local_size Number of elements in the local state vector `u`
   * @param rhs Right-hand side callable, invocable as
   *            `rhs(double t, const std::vector<double>& u,
   *            std::vector<double>& du)`
   *
   * @post All buffers are value-initialized to 0.0
   */
  RK3HeunStepper(double dt, std::size_t local_size, Rhs rhs)
      : m_dt(dt), m_k1(local_size, 0.0), m_k2(local_size, 0.0),
        m_u_temp(local_size, 0.0), m_rhs(std::move(rhs)) {}

  /**
   * @brief Advance `u` by one Heun third-order RK3 step in place; returns
   *        the new time.
   *
   * ## Algorithm
   *
   * 1. **Stage 1**: Evaluate RHS at `(t, u)`
   *    ```
   *    rhs(t, u, m_k1)
   *    ```
   * 2. **Stage 2**: Build the stage-2 evaluation point and evaluate RHS
   *    there
   *    ```
   *    m_u_temp = u + dt/3 * m_k1
   *    rhs(t + dt/3, m_u_temp, m_k2)
   *    ```
   * 3. **Stage 3**: Build the stage-3 evaluation point from `k2` **only**
   *    (no `k1` contribution) and evaluate RHS there, reusing `m_k2` to
   *    store the result since the old `k2` value is no longer needed
   *    ```
   *    m_u_temp = u + 2*dt/3 * m_k2
   *    rhs(t + 2*dt/3, m_u_temp, m_k2)   // m_k2 now holds k3
   *    ```
   * 4. **Combination**: Apply the weighted update (`k2` does not
   *    contribute)
   *    ```
   *    u += dt/4 * m_k1 + dt*3/4 * m_k2   // m_k2 holds k3 here
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
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(u.size());

    // Stage 1: k1 = rhs(t, u)
    m_rhs(t, u, m_k1);

    // Stage 2: k2 = rhs(t + dt/3, u + dt/3 * k1)
    for (std::ptrdiff_t li = 0; li < n; ++li) {
      m_u_temp[static_cast<std::size_t>(li)] =
          u[static_cast<std::size_t>(li)] +
          (m_dt / 3.0) * m_k1[static_cast<std::size_t>(li)];
    }
    m_rhs(t + m_dt / 3.0, m_u_temp, m_k2);

    // Stage 3: k3 = rhs(t + 2*dt/3, u + 2*dt/3 * k2) -- deliberately no k1
    // contribution here, per Heun's third-order tableau. m_k2 is reused to
    // store k3 once its old value has been consumed above.
    for (std::ptrdiff_t li = 0; li < n; ++li) {
      m_u_temp[static_cast<std::size_t>(li)] =
          u[static_cast<std::size_t>(li)] +
          (2.0 * m_dt / 3.0) * m_k2[static_cast<std::size_t>(li)];
    }
    m_rhs(t + 2.0 * m_dt / 3.0, m_u_temp, m_k2); // m_k2 now holds k3

    // Combination: u += dt/4 * k1 + dt*3/4 * k3 -- k2 does not appear here.
    for (std::ptrdiff_t li = 0; li < n; ++li) {
      u[static_cast<std::size_t>(li)] +=
          (m_dt / 4.0) * m_k1[static_cast<std::size_t>(li)] +
          (3.0 * m_dt / 4.0) * m_k2[static_cast<std::size_t>(li)];
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
  double m_dt{0.0};             ///< Time step size
  std::vector<double> m_k1;     ///< k1 = RHS at (t, u); kept for combination
  std::vector<double> m_k2;     ///< k2, then reused in place to hold k3
  std::vector<double> m_u_temp; ///< Staging buffer for stage-2/3 eval points
  Rhs m_rhs;                    ///< Right-hand side callable
};

} // namespace pfc::sim::steppers
