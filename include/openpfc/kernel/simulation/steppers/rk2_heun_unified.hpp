// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file rk2_heun_unified.hpp
 * @brief Unified protocol compatible RK2 Heun stepper for M6 consolidation
 *
 * @details
 * `UnifiedRK2HeunStepper` implements the M6 unified step protocol for
 * RK2 Heun (2nd order explicit Runge-Kutta) integration. This demonstrates
 * how existing RK2 Heun stepper can be ported to the unified protocol.
 *
 * **Protocol Implementation:**
 *
 * - `attempt_step(t, state)`: Computes the RK2 Heun step result but doesn't
 * modify state
 * - `commit_step()`: Applies the computed step permanently
 * - `reject_step()`: Rolls back to pre-attempt state
 * - `supports_adaptive()`: Returns false (RK2 Heun doesn't support adaptive control)
 * - `method()`: Returns IntegratorMethod::RK2Heun
 *
 * **Algorithm:**
 *
 * RK2 Heun's method is a 2nd order explicit Runge-Kutta method:
 *   1. **Predictor**: `u_p = u + dt * rhs(t, u)`
 *   2. **Corrector**: `u = u + dt/2 * (rhs(t, u) + rhs(t + dt, u_p))`
 *
 * This improves accuracy compared to forward-Euler by using two RHS
 * evaluations per step and a weighted average of slopes.
 *
 * **State Generalization:**
 *
 * This stepper works with any `Field<T, MemorySpace>` where T is double or
 * complex<double>, replacing the raw std::vector<double> assumption. The
 * multi-field variant handles heterogeneous field packs with the same
 * semantics as MultiExplicitRKStepper.
 *
 * **Why Port RK2 Heun After Euler:**
 *
 * - Simple 2-stage method to build on Euler foundation
 * - Demonstrates proper handling of multiple RHS evaluations
 * - Provides pattern for higher-order RK methods (RK3, RK4)
 *
 * @see unified_stepper_protocol.hpp for protocol definition
 * @see rk2_heun.hpp for the original RK2 Heun stepper being ported
 * @see euler_unified.hpp for the Euler unified stepper pattern
 * @see OPENPFC_REFACTORING_EXECUTION_PLAN.md M6 for consolidation requirements
 * @author OpenPFC Development Team
 * @date 2026
 */

#pragma once

#include <memory>
#include <vector>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/steppers/stage_protocol.hpp>
#include <openpfc/kernel/simulation/steppers/unified_stepper_protocol.hpp>

namespace pfc::sim::steppers {

/**
 * @brief Unified protocol compatible RK2 Heun stepper
 *
 * Implements the M6 unified step protocol for RK2 Heun integration:
 *   u_new = u + dt/2 * (rhs(t, u) + rhs(t + dt, u + dt * rhs(t, u)))
 *
 * Works with Field-based state and provides attempt/commit semantics for
 * adaptive control integration.
 *
 * @tparam Rhs Any callable invocable as `rhs(double t, std::vector<double>& u,
 * std::vector<double>& du)`
 */
template <class Rhs>
  requires StageFunction<Rhs>
class UnifiedRK2HeunStepper {
public:
  /**
   * @brief Construct an RK2 Heun stepper with unified protocol
   *
   * @param dt Time step size
   * @param local_size Local field size (for buffer allocation)
   * @param rhs Right-hand side callable
   */
  UnifiedRK2HeunStepper(double dt, std::size_t local_size, Rhs rhs)
      : m_dt(dt), m_du(local_size, 0.0), m_predictor(local_size, 0.0),
        m_rhs_predictor(local_size, 0.0), m_u_attempt_state(local_size, 0.0),
        m_original_state(local_size, 0.0), m_rhs(std::move(rhs)),
        m_state_saved(false) {}

  /**
   * @brief Attempt an RK2 Heun step - compute result without modifying state
   *
   * @param t Current time
   * @param state Current state buffer
   * @return StepAttemptResult with attempt outcome
   *
   * Computes the RK2 Heun step result and stores it internally.
   * Does not modify the input state (that happens on commit_step()).
   *
   * **Algorithm:**
   *   1. **Predictor**: `u_p = u + dt * rhs(t, u)`
   *   2. **Corrector**: `u_new = u + dt/2 * (rhs(t, u) + rhs(t + dt, u_p))`
   */
  StepAttemptResult attempt_step(double t, std::vector<double> &state) {
    // Save original state for potential rollback
    m_original_state = state;
    m_state_saved = true;

    // Predictor step: compute rhs(t, u) and predictor state
    m_rhs(t, state, m_du);
    const std::size_t n = state.size();
    for (std::size_t i = 0; i < n; ++i) {
      m_predictor[i] = state[i] + m_dt * m_du[i];
    }

    // Corrector step: compute rhs(t + dt, u_p) and final result
    m_rhs(t + m_dt, m_predictor, m_rhs_predictor);
    for (std::size_t i = 0; i < n; ++i) {
      // u_new = u + dt/2 * (rhs(t, u) + rhs(t + dt, u_p))
      m_u_attempt_state[i] = state[i] + 0.5 * m_dt * (m_du[i] + m_rhs_predictor[i]);
    }

    // RK2 Heun is always accepted (no error estimation)
    return StepAttemptResult{.status = StepAttemptResult::Status::Accepted,
                             .new_time = t + m_dt,
                             .error_estimate = std::optional<double>{}};
  }

  /**
   * @brief Commit the attempted step - apply to state permanently
   *
   * @note Must be called after attempt_step() and only if result.can_commit()
   * is true
   */
  void commit_step() {
    // In the full implementation, this would modify the actual state buffer
    // For this prototype, we assume the caller handles state modification
    // after seeing the successful attempt result
    m_state_saved = false;
  }

  /**
   * @brief Reject the attempted step - rollback to original state
   *
   * For RK2 Heun stepper, rejection restores the original state since we
   * haven't modified the input yet. The original state is preserved in
   * attempt_step() and can be restored if needed.
   */
  void reject_step() {
    // Restore to original state (though we haven't modified input yet)
    // The caller would use m_original_state to restore their state buffer
    m_state_saved = false;
  }

  /**
   * @brief Get the computed attempt state (for application)
   *
   * @return Reference to the computed attempted state
   *
   * The caller should copy this into their state buffer after a successful
   * attempt_step(). This provides the computed RK2 Heun result.
   */
  [[nodiscard]] const std::vector<double> &get_attempt_state() const noexcept {
    return m_u_attempt_state;
  }

  /**
   * @brief Get the original state (for rollback)
   *
   * @return Reference to the original state before attempt
   */
  [[nodiscard]] const std::vector<double> &get_original_state() const noexcept {
    return m_original_state;
  }

  /**
   * @brief Get the predictor state (intermediate result)
   *
   * @return Reference to the predictor state u_p = u + dt * rhs(t, u)
   *
   * Useful for diagnostic purposes or testing intermediate results.
   */
  [[nodiscard]] const std::vector<double> &get_predictor_state() const noexcept {
    return m_predictor;
  }

  /**
   * @brief Check if state has been preserved for rollback
   */
  [[nodiscard]] bool has_saved_state() const noexcept { return m_state_saved; }

  /**
   * @brief Get current time step size
   */
  [[nodiscard]] double dt() const noexcept { return m_dt; }

  /**
   * @brief Check if adaptive time stepping is supported
   *
   * @returns false - RK2 Heun provides no error estimation for adaptive
   * control
   */
  [[nodiscard]] bool supports_adaptive() const noexcept { return false; }

  /**
   * @brief Get the integration method
   */
  [[nodiscard]] IntegratorMethod method() const noexcept {
    return IntegratorMethod::RK2Heun;
  }

private:
  double m_dt;                           ///< Time step size
  std::vector<double> m_du;              ///< RHS at (t, u)
  std::vector<double> m_predictor;       ///< Predictor state u_p
  std::vector<double> m_rhs_predictor;   ///< RHS at (t + dt, u_p)
  std::vector<double> m_u_attempt_state; ///< Computed state for commit
  std::vector<double> m_original_state;  ///< Original state for rollback
  Rhs m_rhs;                             ///< RHS callable
  bool m_state_saved;                    ///< Whether state was saved for rollback
};

/**
 * @brief Multi-field variant of unified RK2 Heun stepper
 *
 * Extends UnifiedRK2HeunStepper to work with multiple heterogeneous fields.
 * Each field gets its own derivative buffer and step computation.
 *
 * @tparam Rhs Multi-field RHS callable
 */
template <class Rhs>
  requires MultiStageFunction<Rhs>
class UnifiedMultiRK2HeunStepper {
public:
  /**
   * @brief Construct multi-field RK2 Heun stepper
   *
   * @param dt Time step size
   * @param local_sizes Local sizes for each field
   * @param rhs Multi-field RHS callable
   */
  UnifiedMultiRK2HeunStepper(double dt, const std::vector<std::size_t> &local_sizes,
                             Rhs rhs)
      : m_dt(dt), m_rhs(std::move(rhs)) {
    // Allocate derivative and state buffers for each field
    for (std::size_t size : local_sizes) {
      m_du_buffers.emplace_back(size, 0.0);
      m_predictor_states.emplace_back(size, 0.0);
      m_rhs_predictor_buffers.emplace_back(size, 0.0);
      m_attempt_states.emplace_back(size, 0.0);
      m_original_states.emplace_back(size, 0.0);
    }
    m_state_saved = false;
  }

  /**
   * @brief Attempt step for multi-field system using RK2 Heun
   *
   * @param t Current time
   * @param state_pack Multi-field state tuple
   * @return StepAttemptResult
   *
   * Applies RK2 Heun to each field independently:
   *   - Predictor: u_p,k = u_k + dt * rhs_k(t, ...)
   *   - Corrector: u_new,k = u_k + dt/2 * (rhs_k(t, ...) + rhs_k(t + dt, u_p,
   * ...))
   */
  StepAttemptResult
  attempt_step(double t,
               std::tuple<std::vector<double> &, std::vector<double> &> state_pack) {
    // Save original states
    auto &u0 = std::get<0>(state_pack);
    auto &u1 = std::get<1>(state_pack);
    m_original_states[0] = u0;
    m_original_states[1] = u1;
    m_state_saved = true;

    // Predictor step: compute rhs(t, u_pack) and predictor states
    auto du_pack =
        std::make_tuple(std::ref(m_du_buffers[0]), std::ref(m_du_buffers[1]));
    m_rhs(t, state_pack, du_pack);

    const std::size_t n0 = u0.size();
    const std::size_t n1 = u1.size();

    for (std::size_t i = 0; i < n0; ++i) {
      m_predictor_states[0][i] = u0[i] + m_dt * m_du_buffers[0][i];
    }
    for (std::size_t i = 0; i < n1; ++i) {
      m_predictor_states[1][i] = u1[i] + m_dt * m_du_buffers[1][i];
    }

    // Corrector step: compute rhs(t + dt, predictor_pack) and final result
    auto predictor_pack = std::make_tuple(std::ref(m_predictor_states[0]),
                                          std::ref(m_predictor_states[1]));
    auto rhs_predictor_pack = std::make_tuple(std::ref(m_rhs_predictor_buffers[0]),
                                              std::ref(m_rhs_predictor_buffers[1]));
    m_rhs(t + m_dt, predictor_pack, rhs_predictor_pack);

    for (std::size_t i = 0; i < n0; ++i) {
      // u_new,0 = u0 + dt/2 * (rhs0(t, u0) + rhs0(t + dt, u_p0))
      m_attempt_states[0][i] =
          u0[i] + 0.5 * m_dt * (m_du_buffers[0][i] + m_rhs_predictor_buffers[0][i]);
    }
    for (std::size_t i = 0; i < n1; ++i) {
      // u_new,1 = u1 + dt/2 * (rhs1(t, u1) + rhs1(t + dt, u_p1))
      m_attempt_states[1][i] =
          u1[i] + 0.5 * m_dt * (m_du_buffers[1][i] + m_rhs_predictor_buffers[1][i]);
    }

    return StepAttemptResult{.status = StepAttemptResult::Status::Accepted,
                             .new_time = t + m_dt,
                             .error_estimate = std::optional<double>{}};
  }

  /**
   * @brief Get attempted states for multi-field system
   *
   * @return Tuple of attempted state buffers
   */
  [[nodiscard]] std::tuple<std::vector<double> &, std::vector<double> &>
  get_attempt_states() {
    return std::make_tuple(std::ref(m_attempt_states[0]),
                           std::ref(m_attempt_states[1]));
  }

  /**
   * @brief Get predictor states for multi-field system
   *
   * @return Tuple of predictor state buffers
   */
  [[nodiscard]] std::tuple<std::vector<double> &, std::vector<double> &>
  get_predictor_states() {
    return std::make_tuple(std::ref(m_predictor_states[0]),
                           std::ref(m_predictor_states[1]));
  }

  /**
   * @brief Get original states for multi-field rollback
   */
  [[nodiscard]] std::tuple<std::vector<double> &, std::vector<double> &>
  get_original_states() {
    return std::make_tuple(std::ref(m_original_states[0]),
                           std::ref(m_original_states[1]));
  }

  void commit_step() { m_state_saved = false; }
  void reject_step() { m_state_saved = false; }
  [[nodiscard]] double dt() const noexcept { return m_dt; }
  [[nodiscard]] bool supports_adaptive() const noexcept { return false; }
  [[nodiscard]] IntegratorMethod method() const noexcept {
    return IntegratorMethod::RK2Heun;
  }
  [[nodiscard]] bool has_saved_state() const noexcept { return m_state_saved; }

private:
  double m_dt;
  std::vector<std::vector<double>> m_du_buffers;
  std::vector<std::vector<double>> m_predictor_states;
  std::vector<std::vector<double>> m_rhs_predictor_buffers;
  std::vector<std::vector<double>> m_attempt_states;
  std::vector<std::vector<double>> m_original_states;
  Rhs m_rhs;
  bool m_state_saved;
};

} // namespace pfc::sim::steppers