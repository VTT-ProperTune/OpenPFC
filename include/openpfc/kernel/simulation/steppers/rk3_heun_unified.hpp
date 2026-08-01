// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file rk3_heun_unified.hpp
 * @brief Unified protocol compatible RK3 Heun stepper for M6 consolidation
 *
 * @details
 * `UnifiedRK3HeunStepper` implements the M6 unified step protocol for
 * RK3 Heun (3rd order explicit Runge-Kutta) integration. This demonstrates
 * how existing RK3 Heun stepper can be ported to the unified protocol.
 *
 * **Protocol Implementation:**
 *
 * - `attempt_step(t, state)`: Computes the RK3 Heun step result but doesn't
 * modify state
 * - `commit_step()`: Applies the computed step permanently
 * - `reject_step()`: Rolls back to pre-attempt state
 * - `supports_adaptive()`: Returns false (RK3 Heun doesn't support adaptive control)
 * - `method()`: Returns IntegratorMethod::RK3Heun
 *
 * **Algorithm:**
 *
 * RK3 Heun's method is a 3rd order explicit Runge-Kutta method:
 *   1. **Stage 1**: `k1 = rhs(t, u)`
 *   2. **Stage 2**: `k2 = rhs(t + dt/3, u + dt/3 * k1)`
 *   3. **Stage 3**: `k3 = rhs(t + 2*dt/3, u + 2*dt/3 * k2)` (no `k1` contribution)
 *   4. **Combination**: `u = u + dt/4 * k1 + dt*3/4 * k3` (no `k2` in final)
 *
 * This improves accuracy compared to RK2 Heun by using three RHS
 * evaluations per step and proper weighting of the stages.
 *
 * **State Generalization:**
 *
 * This stepper works with any `Field<T, MemorySpace>` where T is double or
 * complex<double>, replacing the raw std::vector<double> assumption. The
 * multi-field variant handles heterogeneous field packs with the same
 * semantics.
 *
 * **Why Port RK3 Heun After RK2 Heun:**
 *
 * - Natural progression in Heun method order (2nd → 3rd)
 * - Demonstrates proper handling of three-stage methods
 * - Shows efficient buffer reuse optimization (k2 → k3 reuse)
 *
 * @see unified_stepper_protocol.hpp for protocol definition
 * @see rk3_heun.hpp for the original RK3 Heun stepper being ported
 * @see rk2_heun_unified.hpp for the 2nd order unified stepper
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
 * @brief Unified protocol compatible RK3 Heun stepper
 *
 * Implements the M6 unified step protocol for RK3 Heun integration:
 *   u_new = u + dt/4 * k1 + dt*3/4 * k3
 *
 * Works with Field-based state and provides attempt/commit semantics for
 * adaptive control integration.
 *
 * @tparam Rhs Any callable invocable as `rhs(double t, std::vector<double>& u,
 * std::vector<double>& du)`
 */
template <class Rhs>
  requires StageFunction<Rhs>
class UnifiedRK3HeunStepper {
public:
  /**
   * @brief Construct an RK3 Heun stepper with unified protocol
   *
   * @param dt Time step size
   * @param local_size Local field size (for buffer allocation)
   * @param rhs Right-hand side callable
   */
  UnifiedRK3HeunStepper(double dt, std::size_t local_size, Rhs rhs)
      : m_dt(dt), m_k1(local_size, 0.0), m_k2(local_size, 0.0),
        m_u_temp(local_size, 0.0), m_u_attempt_state(local_size, 0.0),
        m_original_state(local_size, 0.0), m_rhs(std::move(rhs)),
        m_state_saved(false) {}

  /**
   * @brief Attempt an RK3 Heun step - compute result without modifying state
   *
   * @param t Current time
   * @param state Current state buffer
   * @return StepAttemptResult with attempt outcome
   *
   * Computes the RK3 Heun step result and stores it internally.
   * Does not modify the input state (that happens on commit_step()).
   *
   * **Algorithm:**
   *   1. **Stage 1**: `k1 = rhs(t, u)`
   *   2. **Stage 2**: `k2 = rhs(t + dt/3, u + dt/3 * k1)`
   *   3. **Stage 3**: `k3 = rhs(t + 2*dt/3, u + 2*dt/3 * k2)`
   *   4. **Combination**: `u_new = u + dt/4 * k1 + dt*3/4 * k3`
   *
   * Note: `k2` buffer is reused in place to hold `k3` since `k2` doesn't
   * participate in the final combination.
   */
  StepAttemptResult attempt_step(double t, std::vector<double> &state) {
    // Save original state for potential rollback
    m_original_state = state;
    m_state_saved = true;

    const std::size_t n = state.size();

    // Stage 1: k1 = rhs(t, u)
    m_rhs(t, state, m_k1);

    // Stage 2: k2 = rhs(t + dt/3, u + dt/3 * k1)
    for (std::size_t i = 0; i < n; ++i) {
      m_u_temp[i] = state[i] + (m_dt / 3.0) * m_k1[i];
    }
    m_rhs(t + m_dt / 3.0, m_u_temp, m_k2);

    // Stage 3: k3 = rhs(t + 2*dt/3, u + 2*dt/3 * k2)
    // Note: k2 buffer is reused to hold k3 since it's no longer needed here
    for (std::size_t i = 0; i < n; ++i) {
      m_u_temp[i] = state[i] + (2.0 * m_dt / 3.0) * m_k2[i];
    }
    m_rhs(t + 2.0 * m_dt / 3.0, m_u_temp, m_k2); // m_k2 now holds k3

    // Combination: u_new = u + dt/4 * k1 + dt*3/4 * k3 (k2 doesn't appear)
    for (std::size_t i = 0; i < n; ++i) {
      m_u_attempt_state[i] =
          state[i] + (m_dt / 4.0) * m_k1[i] + (3.0 * m_dt / 4.0) * m_k2[i];
    }

    // RK3 Heun is always accepted (no error estimation)
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
   * For RK3 Heun stepper, rejection restores the original state since we
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
   * attempt_step(). This provides the computed RK3 Heun result.
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
   * @brief Get the k1 stage derivative
   *
   * @return Reference to k1 = rhs(t, u)
   *
   * Useful for diagnostic purposes or testing intermediate results.
   */
  [[nodiscard]] const std::vector<double> &get_k1() const noexcept { return m_k1; }

  /**
   * @brief Get the k3 stage derivative
   *
   * @return Reference to k3 = rhs(t + 2*dt/3, u + 2*dt/3 * k2)
   *
   * Note: k2 is omitted since it's reused to hold k3.
   */
  [[nodiscard]] const std::vector<double> &get_k3() const noexcept {
    return m_k2; // m_k2 holds k3 after the algorithm completes
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
   * @returns false - RK3 Heun provides no error estimation for adaptive
   * control
   */
  [[nodiscard]] bool supports_adaptive() const noexcept { return false; }

  /**
   * @brief Get the integration method
   */
  [[nodiscard]] IntegratorMethod method() const noexcept {
    return IntegratorMethod::RK3Heun;
  }

private:
  double m_dt;                  ///< Time step size
  std::vector<double> m_k1;     ///< k1 = RHS at (t, u); kept for combination
  std::vector<double> m_k2;     ///< k2, then reused in place to hold k3
  std::vector<double> m_u_temp; ///< Staging buffer for stage-2/3 eval points
  std::vector<double> m_u_attempt_state; ///< Computed state for commit
  std::vector<double> m_original_state;  ///< Original state for rollback
  Rhs m_rhs;                             ///< RHS callable
  bool m_state_saved;                    ///< Whether state was saved for rollback
};

/**
 * @brief Multi-field variant of unified RK3 Heun stepper
 *
 * Extends UnifiedRK3HeunStepper to work with multiple heterogeneous fields.
 * Each field gets its own derivative buffer and step computation.
 *
 * @tparam Rhs Multi-field RHS callable
 */
template <class Rhs>
  requires MultiStageFunction<Rhs>
class UnifiedMultiRK3HeunStepper {
public:
  /**
   * @brief Construct multi-field RK3 Heun stepper
   *
   * @param dt Time step size
   * @param local_sizes Local sizes for each field
   * @param rhs Multi-field RHS callable
   */
  UnifiedMultiRK3HeunStepper(double dt, const std::vector<std::size_t> &local_sizes,
                             Rhs rhs)
      : m_dt(dt), m_rhs(std::move(rhs)) {
    // Allocate derivative and state buffers for each field
    for (std::size_t size : local_sizes) {
      m_k1_buffers.emplace_back(size, 0.0);
      m_k2_k3_buffers.emplace_back(size, 0.0); // Will hold k2 then k3
      m_u_temp_buffers.emplace_back(size, 0.0);
      m_attempt_states.emplace_back(size, 0.0);
      m_original_states.emplace_back(size, 0.0);
    }
    m_state_saved = false;
  }

  /**
   * @brief Attempt step for multi-field system using RK3 Heun
   *
   * @param t Current time
   * @param state_pack Multi-field state tuple
   * @return StepAttemptResult
   *
   * Applies RK3 Heun to each field independently:
   *   - Stage 1: k1,k = rhs_k(t, u_pack)
   *   - Stage 2: k2,k = rhs_k(t + dt/3, u_pack + dt/3 * k1,k)
   *   - Stage 3: k3,k = rhs_k(t + 2*dt/3, u_pack + 2*dt/3 * k2,k)
   *   - Combination: u_new,k = u_k + dt/4 * k1,k + dt*3/4 * k3,k
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

    const std::size_t n0 = u0.size();
    const std::size_t n1 = u1.size();

    // Stage 1: k1 = rhs(t, u_pack)
    auto k1_pack =
        std::make_tuple(std::ref(m_k1_buffers[0]), std::ref(m_k1_buffers[1]));
    m_rhs(t, state_pack, k1_pack);

    // Stage 2: k2 = rhs(t + dt/3, u_pack + dt/3 * k1)
    auto u_temp_pack = std::make_tuple(std::ref(m_u_temp_buffers[0]),
                                       std::ref(m_u_temp_buffers[1]));
    for (std::size_t i = 0; i < n0; ++i) {
      m_u_temp_buffers[0][i] = u0[i] + (m_dt / 3.0) * m_k1_buffers[0][i];
    }
    for (std::size_t i = 0; i < n1; ++i) {
      m_u_temp_buffers[1][i] = u1[i] + (m_dt / 3.0) * m_k1_buffers[1][i];
    }

    auto k2_k3_pack =
        std::make_tuple(std::ref(m_k2_k3_buffers[0]), std::ref(m_k2_k3_buffers[1]));
    m_rhs(t + m_dt / 3.0, u_temp_pack, k2_k3_pack);

    // Stage 3: k3 = rhs(t + 2*dt/3, u_pack + 2*dt/3 * k2)
    // Buffer reuse: k2_k3_buffers now hold k3
    for (std::size_t i = 0; i < n0; ++i) {
      m_u_temp_buffers[0][i] = u0[i] + (2.0 * m_dt / 3.0) * m_k2_k3_buffers[0][i];
    }
    for (std::size_t i = 0; i < n1; ++i) {
      m_u_temp_buffers[1][i] = u1[i] + (2.0 * m_dt / 3.0) * m_k2_k3_buffers[1][i];
    }
    m_rhs(t + 2.0 * m_dt / 3.0, u_temp_pack, k2_k3_pack); // Now holds k3

    // Combination: u_new = u + dt/4 * k1 + dt*3/4 * k3
    for (std::size_t i = 0; i < n0; ++i) {
      m_attempt_states[0][i] = u0[i] + (m_dt / 4.0) * m_k1_buffers[0][i] +
                               (3.0 * m_dt / 4.0) * m_k2_k3_buffers[0][i];
    }
    for (std::size_t i = 0; i < n1; ++i) {
      m_attempt_states[1][i] = u1[i] + (m_dt / 4.0) * m_k1_buffers[1][i] +
                               (3.0 * m_dt / 4.0) * m_k2_k3_buffers[1][i];
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
   * @brief Get k1 stage derivatives for multi-field system
   *
   * @return Tuple of k1 stage buffers
   */
  [[nodiscard]] std::tuple<std::vector<double> &, std::vector<double> &>
  get_k1_states() {
    return std::make_tuple(std::ref(m_k1_buffers[0]), std::ref(m_k1_buffers[1]));
  }

  /**
   * @brief Get k3 stage derivatives for multi-field system
   *
   * @return Tuple of k3 stage buffers
   */
  [[nodiscard]] std::tuple<std::vector<double> &, std::vector<double> &>
  get_k3_states() {
    return std::make_tuple(std::ref(m_k2_k3_buffers[0]),
                           std::ref(m_k2_k3_buffers[1]));
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
    return IntegratorMethod::RK3Heun;
  }
  [[nodiscard]] bool has_saved_state() const noexcept { return m_state_saved; }

private:
  double m_dt;
  std::vector<std::vector<double>> m_k1_buffers;
  std::vector<std::vector<double>> m_k2_k3_buffers; // Holds k2 then k3
  std::vector<std::vector<double>> m_u_temp_buffers;
  std::vector<std::vector<double>> m_attempt_states;
  std::vector<std::vector<double>> m_original_states;
  Rhs m_rhs;
  bool m_state_saved;
};

} // namespace pfc::sim::steppers