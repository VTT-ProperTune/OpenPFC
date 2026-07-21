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

#include <openpfc/kernel/simulation/steppers/butcher_tableau.hpp>
#include <openpfc/kernel/simulation/steppers/stage_protocol.hpp>

namespace pfc::sim::steppers {

/**
 * @brief Evidence returned by one embedded RK step attempt.
 *
 * Views bind to method-owned buffers and remain valid until the next
 * `attempt` call or stepper destruction. `success` is computational
 * completion only — not an adaptive accept/reject decision.
 */
struct EmbeddedStepAttemptResult {
  double t0{};
  double dt{};
  double t1{}; ///< `t0 + dt` on success
  bool success{false};
  unsigned int rhs_evals{0}; ///< equals `tableau.stage_count()` on success
  const std::vector<double> &u_high;
  const std::vector<double> &u_low;
  const std::vector<double> &error; ///< elementwise `u_high - u_low`

  EmbeddedStepAttemptResult(double t0_in, double dt_in, double t1_in,
                            bool success_in, unsigned int rhs_evals_in,
                            const std::vector<double> &u_high_in,
                            const std::vector<double> &u_low_in,
                            const std::vector<double> &error_in)
      : t0(t0_in), dt(dt_in), t1(t1_in), success(success_in),
        rhs_evals(rhs_evals_in), u_high(u_high_in), u_low(u_low_in),
        error(error_in) {}
};

/**
 * @brief CPU embedded explicit RK step-attempt stepper.
 *
 * Requires `tableau.has_embedded()`. Shared stages are evaluated once; dual
 * accumulation with `b` and `b_hat` fills method-owned candidate/error
 * storage. Does not decide accept/reject or next `dt` — that policy stays
 * with the adaptive driver/controller.
 *
 * @tparam Rhs Callable invocable as `rhs(t, u, du)` filling `du`
 *             (`StageFunction`).
 */
template <class Rhs>
  requires StageFunction<Rhs>
class EmbeddedRKStepper {
public:
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
      : m_local_size(local_size), m_du(local_size, 0.0),
        m_u_temp(local_size, 0.0), m_u_high(local_size, 0.0),
        m_u_low(local_size, 0.0), m_error(local_size, 0.0),
        m_tableau(std::move(tableau)), m_rhs(std::move(rhs)) {
    if (!m_tableau.has_embedded()) {
      throw std::invalid_argument(
          "EmbeddedRKStepper requires an embedded ButcherTableau "
          "(has_embedded() == true; missing b_hat / embedded weights)");
    }
    const unsigned int s = m_tableau.stage_count();
    m_k.resize(s);
    for (unsigned int i = 0; i < s; ++i) {
      m_k[i].assign(local_size, 0.0);
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
  [[nodiscard]] EmbeddedStepAttemptResult
  attempt(double t, double dt, const std::vector<double> &u) {
    if (u.size() != m_local_size) {
      throw std::invalid_argument(
          "EmbeddedRKStepper::attempt: u.size() (" +
          std::to_string(u.size()) + ") != local_size (" +
          std::to_string(m_local_size) + ")");
    }

    const unsigned int s = m_tableau.stage_count();
    unsigned int rhs_evals = 0;

    for (unsigned int i = 0; i < s; ++i) {
      m_u_temp = u;
      for (unsigned int j = 0; j < i; ++j) {
        const double a_ij = m_tableau.a(i, j);
        if (a_ij != 0.0) {
          for (std::size_t idx = 0; idx < m_local_size; ++idx) {
            m_u_temp[idx] += dt * a_ij * m_k[j][idx];
          }
        }
      }

      const double stage_time = t + m_tableau.c(i) * dt;
      m_rhs(stage_time, m_u_temp, m_du);
      m_k[i] = m_du;
      ++rhs_evals;
    }

    m_u_high = u;
    m_u_low = u;
    for (unsigned int i = 0; i < s; ++i) {
      const double b_i = m_tableau.b(i);
      const double b_hat_i = m_tableau.b_hat(i);
      if (b_i != 0.0 || b_hat_i != 0.0) {
        for (std::size_t idx = 0; idx < m_local_size; ++idx) {
          const double k_val = m_k[i][idx];
          if (b_i != 0.0) {
            m_u_high[idx] += dt * b_i * k_val;
          }
          if (b_hat_i != 0.0) {
            m_u_low[idx] += dt * b_hat_i * k_val;
          }
        }
      }
    }

    for (std::size_t idx = 0; idx < m_local_size; ++idx) {
      m_error[idx] = m_u_high[idx] - m_u_low[idx];
    }

    return EmbeddedStepAttemptResult(t, dt, t + dt, true, rhs_evals, m_u_high,
                                     m_u_low, m_error);
  }

  [[nodiscard]] const ButcherTableau<double> &tableau() const noexcept {
    return m_tableau;
  }

  [[nodiscard]] const std::vector<double> &u_high() const noexcept {
    return m_u_high;
  }

  [[nodiscard]] const std::vector<double> &u_low() const noexcept {
    return m_u_low;
  }

  [[nodiscard]] const std::vector<double> &error() const noexcept {
    return m_error;
  }

private:
  std::size_t m_local_size{0};
  std::vector<double> m_du;
  std::vector<std::vector<double>> m_k;
  std::vector<double> m_u_temp;
  std::vector<double> m_u_high;
  std::vector<double> m_u_low;
  std::vector<double> m_error;
  ButcherTableau<double> m_tableau;
  Rhs m_rhs;
};

} // namespace pfc::sim::steppers
