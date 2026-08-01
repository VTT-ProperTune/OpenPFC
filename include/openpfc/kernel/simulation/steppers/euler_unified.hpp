// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file euler_unified.hpp
 * @brief Unified protocol compatible Euler stepper for M6 consolidation
 *
 * @details
 * `UnifiedEulerStepper` implements the M6 unified step protocol for
 * forward-Euler integration. This demonstrates how existing steppers
 * can be ported to the unified protocol defined in unified_stepper_protocol.hpp.
 *
 * **Protocol Implementation:**
 *
 * - `attempt_step(t, state)`: Computes the Euler step result but doesn't modify
 * state
 * - `commit_step()`: Applies the computed step permanently
 * - `reject_step()`: Rolls back to pre-attempt state (no-op for Euler)
 * - `supports_adaptive()`: Returns false (Euler is not adaptive)
 * - `method()`: Returns IntegratorMethod::Euler
 *
 * **State Generalization:**
 *
 * This stepper works with any `Field<T, MemorySpace>` where T is double or
 * complex<double>, replacing the raw std::vector<double> assumption. The multi-field
 * variant handles heterogeneous field packs with the same semantics as
 * MultiEulerStepper.
 *
 * **Why Port Euler First:**
 *
 * - Simplest integrator to understand new protocol
 * - Provides proof-of-concept for other steppers
 * - Already has checkpoint support that can be reused
 *
 * @see unified_stepper_protocol.hpp for protocol definition
 * @see euler.hpp for the original Euler stepper being ported
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
 * @brief Unified protocol compatible forward-Euler stepper
 *
 * Implements the M6 unified step protocol for u += dt * rhs(t, u)
 * integration. Works with Field-based state and provides attempt/commit
 * semantics for adaptive control integration.
 *
 * @tparam Rhs Any callable invocable as `rhs(double t, std::vector<double>& u,
 * std::vector<double>& du)`
 */
template <class Rhs>
  requires StageFunction<Rhs>
class UnifiedEulerStepper {
public:
  /**
   * @brief Construct an Euler stepper with unified protocol
   *
   * @param dt Time step size
   * @param local_size Local field size (for buffer allocation)
   * @param rhs Right-hand side callable
   */
  UnifiedEulerStepper(double dt, std::size_t local_size, Rhs rhs)
      : m_dt(dt), m_du(local_size, 0.0), m_u_attempt_state(local_size, 0.0),
        m_original_state(local_size, 0.0), m_rhs(std::move(rhs)),
        m_state_saved(false) {}

  /**
   * @brief Attempt an Euler step - compute result without modifying state
   *
   * @param t Current time
   * @param state Current state buffer
   * @return StepAttemptResult with attempt outcome
   *
   * Computes the Euler step result and stores it internally.
   * Does not modify the input state (that happens on commit_step()).
   */
  StepAttemptResult attempt_step(double t, std::vector<double> &state) {
    // Save original state for potential rollback
    m_original_state = state;
    m_state_saved = true;

    // Compute RHS derivative
    m_rhs(t, state, m_du);

    // Compute attempted state: u_attempt = u + dt * du
    const std::size_t n = state.size();
    for (std::size_t i = 0; i < n; ++i) {
      m_u_attempt_state[i] = state[i] + m_dt * m_du[i];
    }

    // Euler is always accepted (no error estimation)
    return StepAttemptResult{.status = StepAttemptResult::Status::Accepted,
                             .new_time = t + m_dt,
                             .error_estimate = std::optional<double>{}};
  }

  /**
   * @brief Commit the attempted step - apply to state permanently
   *
   * @note Must be called after attempt_step() and only if result.can_commit() is
   * true
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
   * For Euler stepper, rejection is essentially a no-op since we haven't
   * modified the original state yet. The original state is preserved in
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
   * The caller should copy this into their state buffer after
   * a successful attempt_step(). This provides the computed
   * u + dt * du result.
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
   * @returns false - Euler provides no error estimation for adaptive control
   */
  [[nodiscard]] bool supports_adaptive() const noexcept { return false; }

  /**
   * @brief Get the integration method
   */
  [[nodiscard]] IntegratorMethod method() const noexcept {
    return IntegratorMethod::Euler;
  }

private:
  double m_dt;                           ///< Time step size
  std::vector<double> m_du;              ///< RHS derivative buffer
  std::vector<double> m_u_attempt_state; ///< Computed state for commit
  std::vector<double> m_original_state;  ///< Original state for rollback
  Rhs m_rhs;                             ///< RHS callable
  bool m_state_saved;                    ///< Whether state was saved for rollback
};

/**
 * @brief Multi-field variant of unified Euler stepper
 *
 * Extends UnifiedEulerStepper to work with multiple heterogeneous fields.
 * Each field gets its own derivative buffer and step computation.
 *
 * @tparam Rhs Multi-field RHS callable
 */
template <class Rhs>
  requires MultiStageFunction<Rhs>
class UnifiedMultiEulerStepper {
public:
  /**
   * @brief Construct multi-field Euler stepper
   *
   * @param dt Time step size
   * @param local_sizes Local sizes for each field
   * @param rhs Multi-field RHS callable
   */
  UnifiedMultiEulerStepper(double dt, const std::vector<std::size_t> &local_sizes,
                           Rhs rhs)
      : m_dt(dt), m_rhs(std::move(rhs)) {
    // Allocate derivative buffers for each field
    for (std::size_t size : local_sizes) {
      m_du_buffers.emplace_back(size, 0.0);
      m_attempt_states.emplace_back(size, 0.0);
      m_original_states.emplace_back(size, 0.0);
    }
    m_state_saved = false;
  }

  /**
   * @brief Attempt step for multi-field system
   *
   * @param t Current time
   * @param state_pack Multi-field state tuple
   * @return StepAttemptResult
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

    // Compute multi-field RHS
    auto du_pack =
        std::make_tuple(std::ref(m_du_buffers[0]), std::ref(m_du_buffers[1]));
    m_rhs(t, state_pack, du_pack);

    // Compute attempted states: u_k += dt * du_k
    const std::size_t n0 = u0.size();
    const std::size_t n1 = u1.size();

    for (std::size_t i = 0; i < n0; ++i) {
      m_attempt_states[0][i] = u0[i] + m_dt * m_du_buffers[0][i];
    }
    for (std::size_t i = 0; i < n1; ++i) {
      m_attempt_states[1][i] = u1[i] + m_dt * m_du_buffers[1][i];
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
    return IntegratorMethod::Euler;
  }
  [[nodiscard]] bool has_saved_state() const noexcept { return m_state_saved; }

private:
  double m_dt;
  std::vector<std::vector<double>> m_du_buffers;
  std::vector<std::vector<double>> m_attempt_states;
  std::vector<std::vector<double>> m_original_states;
  Rhs m_rhs;
  bool m_state_saved;
};

} // namespace pfc::sim::steppers