// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file embedded_rk.hpp
 * @brief Embedded explicit Runge-Kutta step-attempt API (high/low + error).
 *
 * @details
 * `EmbeddedRKStepper` evaluates shared explicit stages once from an embedded
 * `ButcherTableau` and exposes isolated high-order (`u_high`), low-order
 * embedded (`u_low`), and error-difference (`error = u_high - u_low`) buffers.
 *
 * The accepted input state is never mutated. Adaptive accept/reject and next
 * `dt` selection remain driver/controller-owned — `success` means only that
 * the attempt completed computationally (stages evaluated, candidates formed).
 * `attempt` returns `StepAttemptResult` with `candidate == u_high()`. Low-order
 * state, error difference, and RHS-eval count stay on stepper accessors.
 *
 * FSAL stage reuse is intentionally out of scope for this slice. If a future
 * cache is added under the same leaf, it must be valid only after accepted
 * steps and invalidated on reject, restart, or configuration change.
 *
 * @see butcher_tableau.hpp for embedded coefficient factories
 * @see explicit_rk.hpp for the fixed-step in-place stepper (orthogonal API)
 */

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/steppers/butcher_tableau.hpp>
#include <openpfc/kernel/simulation/steppers/stage_protocol.hpp>
#include <openpfc/kernel/simulation/steppers/step_attempt.hpp>

namespace pfc::sim::steppers {

/**
 * @brief CPU embedded explicit RK step-attempt stepper.
 *
 * Requires `tableau.has_embedded()`. Shared stages are evaluated once; dual
 * accumulation with `b` and `b_hat` fills method-owned candidate/error
 * storage. Does not decide accept/reject or next `dt` — that policy stays
 * with the adaptive driver/controller.
 *
 * @tparam Rhs    Callable invocable as `rhs(t, u, du)` filling `du`
 *                (`StageFunctionFor<Rhs, Scalar>`).
 * @tparam Scalar Field element type (`double` or `std::complex<double>`).
 */
template <class Rhs, class Scalar = double>
  requires StageFunctionFor<Rhs, Scalar>
class EmbeddedRKStepper {
public:
  using scalar_type = Scalar;
  using Attempt = StepAttempt<Scalar>;

  /**
   * @brief Construct an embedded RK stepper.
   *
   * @param local_size Number of cells in the rank-local field buffer.
   * @param tableau Embedded Butcher tableau (`has_embedded()` must be true).
   * @param rhs RHS callable.
   *
   * @throws std::invalid_argument if `!tableau.has_embedded()`.
   */
  EmbeddedRKStepper(std::size_t local_size, ButcherTableau<double> tableau,
                    Rhs rhs)
      : m_local_size(local_size), m_du(local_size, Scalar{}),
        m_u_temp(local_size, Scalar{}), m_u_high(local_size, Scalar{}),
        m_u_low(local_size, Scalar{}), m_error(local_size, Scalar{}),
        m_tableau(std::move(tableau)), m_rhs(std::move(rhs)) {
    if (!m_tableau.has_embedded()) {
      throw std::invalid_argument(
          "EmbeddedRKStepper requires an embedded ButcherTableau "
          "(has_embedded() == true; missing b_hat / embedded weights)");
    }
    const unsigned int s = m_tableau.stage_count();
    m_k.resize(s);
    for (unsigned int i = 0; i < s; ++i) {
      m_k[i].assign(local_size, Scalar{});
    }
  }

  /**
   * @brief Attempt one embedded RK step without mutating accepted state.
   *
   * Evaluates `stage_count` RHS calls, then forms isolated `u_high`, `u_low`,
   * and `error = u_high - u_low`. Does **not** accept/reject the step or
   * choose the next `dt` — adaptive policy remains driver/controller-owned.
   *
   * @param t Current accepted time.
   * @param dt Proposed step size for this attempt.
   * @param u Accepted state (read-only; never written).
   * @return Attempt evidence with views into method-owned buffers.
   *
   * @throws std::invalid_argument if `u.size() != local_size`.
   */
  [[nodiscard]] Attempt attempt(double t, double dt,
                                const std::vector<Scalar> &u) {
    if (u.size() != m_local_size) {
      throw std::invalid_argument(
          "EmbeddedRKStepper::attempt: u.size() (" +
          std::to_string(u.size()) + ") != local_size (" +
          std::to_string(m_local_size) + ")");
    }

    const unsigned int s = m_tableau.stage_count();
    m_last_rhs_evals = 0;

    for (unsigned int i = 0; i < s; ++i) {
      m_u_temp = u;
      for (unsigned int j = 0; j < i; ++j) {
        const double a_ij = m_tableau.a(i, j);
        if (a_ij != 0.0) {
          const Scalar scale = Scalar(dt * a_ij);
          for (std::size_t idx = 0; idx < m_local_size; ++idx) {
            m_u_temp[idx] += scale * m_k[j][idx];
          }
        }
      }

      const double stage_time = t + m_tableau.c(i) * dt;
      m_rhs(stage_time, m_u_temp, m_du);
      m_k[i] = m_du;
      ++m_last_rhs_evals;
    }

    m_u_high = u;
    m_u_low = u;
    for (unsigned int i = 0; i < s; ++i) {
      const double b_i = m_tableau.b(i);
      const double b_hat_i = m_tableau.b_hat(i);
      if (b_i != 0.0 || b_hat_i != 0.0) {
        const Scalar scale_b = Scalar(dt * b_i);
        const Scalar scale_hat = Scalar(dt * b_hat_i);
        for (std::size_t idx = 0; idx < m_local_size; ++idx) {
          const Scalar k_val = m_k[i][idx];
          if (b_i != 0.0) {
            m_u_high[idx] += scale_b * k_val;
          }
          if (b_hat_i != 0.0) {
            m_u_low[idx] += scale_hat * k_val;
          }
        }
      }
    }

    for (std::size_t idx = 0; idx < m_local_size; ++idx) {
      m_error[idx] = m_u_high[idx] - m_u_low[idx];
    }

    return Attempt(t, dt, t + dt, /*success=*/true, m_u_high);
  }

  /** Isolate high/low/error from a host `Field<Scalar>` (via `vec()`). */
  [[nodiscard]] Attempt attempt(double t, double dt,
                                const pfc::data::Field<Scalar> &u) {
    return attempt(t, dt, u.vec());
  }

  [[nodiscard]] const ButcherTableau<double> &tableau() const noexcept {
    return m_tableau;
  }

  [[nodiscard]] const std::vector<Scalar> &u_high() const noexcept {
    return m_u_high;
  }

  [[nodiscard]] const std::vector<Scalar> &u_low() const noexcept {
    return m_u_low;
  }

  [[nodiscard]] const std::vector<Scalar> &error() const noexcept {
    return m_error;
  }

  /** Number of RHS evaluations in the last successful `attempt`. */
  [[nodiscard]] unsigned int last_rhs_evals() const noexcept {
    return m_last_rhs_evals;
  }

private:
  std::size_t m_local_size{0};
  std::vector<Scalar> m_du;
  std::vector<std::vector<Scalar>> m_k;
  std::vector<Scalar> m_u_temp;
  std::vector<Scalar> m_u_high;
  std::vector<Scalar> m_u_low;
  std::vector<Scalar> m_error;
  unsigned int m_last_rhs_evals{0};
  ButcherTableau<double> m_tableau;
  Rhs m_rhs;
};

} // namespace pfc::sim::steppers
