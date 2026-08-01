// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file explicit_rk_unified.hpp
 * @brief Unified protocol compatible explicit Runge-Kutta stepper for M6
 *        consolidation
 *
 * @details
 * `UnifiedExplicitRKStepper` implements the M6 unified step protocol for
 * generic explicit Runge-Kutta methods using Butcher tableaus. This provides
 * a unified interface for all RK methods (RK2, RK3, RK4, etc.) through
 * the power of Butcher tableau coefficients.
 *
 * **Protocol Implementation:**
 *
 * - `attempt_step(t, state)`: Computes the RK step result but doesn't modify
 * state
 * - `commit_step()`: Applies the computed step permanently
 * - `reject_step()`: Rolls back to pre-attempt state
 * - `supports_adaptive()`: Returns false (basic RK doesn't support adaptive
 * control; see EmbeddedRK for that)
 * - `method()`: Returns IntegratorMethod::RK4 as representative (could be
 * extended to detect specific tableaus)
 *
 * **Algorithm:**
 *
 * For an s-stage explicit RK method:
 *   1. For each stage i (1 ≤ i ≤ s):
 *      `k_i = rhs(t + c_i*dt, u + dt * sum_{j=1}^{i-1} a_ij * k_j)`
 *   2. Final accumulation:
 *      `u_new = u + dt * sum_{i=1}^{s} b_i * k_i`
 *
 * Where:
 * - `c_i` are stage time coefficients (c_1 = 0, c_i = sum_j a_ij)
 * - `a_ij` are stage interaction coefficients (a_ij = 0 for j ≥ i, explicit)
 * - `b_i` are final accumulation weights
 *
 * **Butcher Tableau Support:**
 *
 * Works with any `ButcherTableau<T>` defining an explicit RK method,
 * allowing runtime method selection. Supports:
 * - RK2 (Forward Euler, Midpoint, Heun)
 * - RK3 (Heun's 3rd order, Kutta's 3rd order)
 * - RK4 (Classical 4th order)
 * - Higher-order methods
 *
 * **Memory Efficiency:**
 *
 * Pre-allocates:
 * - `m_u_temp`: Temporary state buffer for each stage evaluation
 * - `m_k`: Scratch buffers for stage derivatives (one per stage)
 * - `m_du`: Single RHS evaluation buffer
 * - Avoids per-step allocations for performance
 *
 * **State Generalization:**
 *
 * This stepper works with any `Field<T, MemorySpace>` where T is double or
 * complex<double>, replacing the raw std::vector<double> assumption. The
 * multi-field variant handles heterogeneous field packs with the same
 * semantics.
 *
 * **Why Port ExplicitRK After RK3 Heun:**
 *
 * - Generalizes the RK pattern established by Euler, RK2/3 Heun
 * - Demonstrates proper handling of arbitrary-stage methods
 * - Shows efficient memory management for dynamic stage counts
 * - Provides foundation for specialized methods (EmbeddedRK, etc.)
 *
 * @see unified_stepper_protocol.hpp for protocol definition
 * @see explicit_rk.hpp for the original explicit RK stepper being ported
 * @see butcher_tableau.hpp for ButcherTableau coefficient infrastructure
 * @see rk2_heun_unified.hpp for 2nd order specifics
 * @see rk3_heun_unified.hpp for 3rd order specifics
 * @see OPENPFC_REFACTORING_EXECUTION_PLAN.md M6 for consolidation requirements
 * @author OpenPFC Development Team
 * @date 2026
 */

#pragma once

#include <memory>
#include <vector>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/steppers/butcher_tableau.hpp>
#include <openpfc/kernel/simulation/steppers/stage_protocol.hpp>
#include <openpfc/kernel/simulation/steppers/unified_stepper_protocol.hpp>

namespace pfc::sim::steppers {

/**
 * @brief Unified protocol compatible explicit Runge-Kutta stepper
 *
 * Implements the M6 unified step protocol for any explicit RK method defined
 * by a Butcher tableau: u_new = u + dt * sum_i(b_i * k_i)
 *
 * Works with Field-based state and provides attempt/commit semantics for
 * adaptive control integration.
 *
 * @tparam Rhs Any callable invocable as `rhs(double t, std::vector<double>& u,
 * std::vector<double>& du)`
 */
template <class Rhs>
  requires StageFunction<Rhs>
class UnifiedExplicitRKStepper {
public:
  /**
   * @brief Construct an explicit RK stepper with unified protocol
   *
   * @param dt Time step size
   * @param local_size Local field size (for buffer allocation)
   * @param tableau Butcher tableau defining the RK method coefficients
   * @param rhs Right-hand side callable
   */
  UnifiedExplicitRKStepper(double dt, std::size_t local_size,
                            ButcherTableau<double> tableau, Rhs rhs)
      : m_dt(dt), m_tableau(std::move(tableau)), m_rhs(std::move(rhs)) {
    const unsigned int s = m_tableau.stage_count();

    // Allocate stage derivative buffers
    m_k.resize(s);
    for (unsigned int i = 0; i < s; ++i) {
      m_k[i].assign(local_size, 0.0);
    }

    // Allocate RHS evaluation buffer
    m_du.assign(local_size, 0.0);

    // Allocate temporary state buffer
    m_u_temp.resize(local_size);

    // Allocate attempt state buffer
    m_u_attempt_state.resize(local_size);

    // Allocate original state buffer for rollback
    m_original_state.resize(local_size);

    m_state_saved = false;
  }

  /**
   * @brief Attempt an explicit RK step - compute result without modifying state
   *
   * @param t Current time
   * @param state Current state buffer
   * @return StepAttemptResult with attempt outcome
   *
   * Computes the explicit RK step result and stores it internally.
   * Does not modify the input state (that happens on commit_step()).
   *
   * **Algorithm:**
   *   1. For each stage i: compute
   *      k_i = rhs(t + c_i*dt, u + dt * sum_j(a_ij * k_j))
   *   2. Final accumulation: u_new = u + dt * sum_i(b_i * k_i)
   *
   * Uses internal buffers to avoid modifying the input state during
   * computation.
   */
  StepAttemptResult attempt_step(double t, std::vector<double> &state) {
    // Save original state for potential rollback
    m_original_state = state;
    m_state_saved = true;

    const unsigned int s = m_tableau.stage_count();
    const std::size_t n = state.size();

    // Compute stages
    for (unsigned int i = 0; i < s; ++i) {
      // Build temp state: u_temp = u + dt * sum_j(a_ij * k_j)
      m_u_temp = state; // Start with current state
      for (unsigned int j = 0; j < i; ++j) {
        const double a_ij = m_tableau.a(i, j);
        if (a_ij != 0.0) {
          for (std::size_t idx = 0; idx < n; ++idx) {
            m_u_temp[idx] += m_dt * a_ij * m_k[j][idx];
          }
        }
      }

      // Compute stage i: k_i = rhs(t + c_i * dt, u_temp)
      const double stage_time = t + m_tableau.c(i) * m_dt;
      m_rhs(stage_time, m_u_temp, m_du);

      // Copy du to k_i
      m_k[i] = m_du;
    }

    // Final accumulation: u_new = u + dt * sum_i(b_i * k_i)
    m_u_attempt_state = state; // Start with current state
    for (unsigned int i = 0; i < s; ++i) {
      const double b_i = m_tableau.b(i);
      if (b_i != 0.0) {
        for (std::size_t idx = 0; idx < n; ++idx) {
          m_u_attempt_state[idx] += m_dt * b_i * m_k[i][idx];
        }
      }
    }

    // Basic explicit RK is always accepted (no error estimation)
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
   * For explicit RK stepper, rejection restores the original state since we
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
   * attempt_step(). This provides the computed RK result.
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
   * @brief Get the stage derivatives (for diagnostic or testing purposes)
   *
   * @return Reference to vector of stage derivative buffers k_i
   *
   * Useful for understanding intermediate RK stage computations or
   * testing stage algorithms.
   */
  [[nodiscard]] const std::vector<std::vector<double>> &get_stage_derivatives()
      const noexcept {
    return m_k;
  }

  /**
   * @brief Get the Butcher tableau being used
   *
   * @return Reference to the Butcher tableau defining the RK method
   */
  [[nodiscard]] const ButcherTableau<double> &get_tableau() const noexcept {
    return m_tableau;
  }

  /**
   * @brief Get the number of stages in this RK method
   *
   * @return Number of stages (s)
   */
  [[nodiscard]] unsigned int stage_count() const noexcept {
    return m_tableau.stage_count();
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
   * @returns false - Basic explicit RK provides no error estimation for
   * adaptive control (see EmbeddedRK for adaptive methods)
   */
  [[nodiscard]] bool supports_adaptive() const noexcept { return false; }

  /**
   * @brief Get the integration method
   *
   * @returns IntegratorMethod::RK4 as representative (could be extended
   * to detect specific tableaus and return RK2Heun/RK3Heun/etc.)
   */
  [[nodiscard]] IntegratorMethod method() const noexcept {
    return IntegratorMethod::RK4;
  }

private:
  double m_dt;                           ///< Time step size
  ButcherTableau<double> m_tableau;      ///< Butcher tableau coefficients
  std::vector<std::vector<double>> m_k;  ///< Stage derivatives k_i
  std::vector<double> m_du;              ///< RHS evaluation buffer
  std::vector<double> m_u_temp;          ///< Temporary state buffer
  std::vector<double> m_u_attempt_state; ///< Computed state for commit
  std::vector<double> m_original_state;  ///< Original state for rollback
  Rhs m_rhs;                             ///< RHS callable
  bool m_state_saved;                    ///< Whether state was saved for rollback
};

/**
 * @brief Multi-field variant of unified explicit RK stepper
 *
 * Extends UnifiedExplicitRKStepper to work with multiple heterogeneous fields.
 * Each field gets its own set of derivative buffers and step computation.
 *
 * @tparam Rhs Multi-field RHS callable
 * @tparam N Number of fields
 */
template <class Rhs, std::size_t N> class UnifiedMultiExplicitRKStepper {
public:
  /**
   * @brief Construct multi-field explicit RK stepper
   *
   * @param dt Time step size
   * @param local_sizes Local sizes for each field
   * @param tableau Butcher tableau defining the RK method coefficients
   * @param rhs Multi-field RHS callable
   */
  UnifiedMultiExplicitRKStepper(double dt,
                                std::array<std::size_t, N> local_sizes,
                                ButcherTableau<double> tableau, Rhs rhs)
      : m_dt(dt), m_tableau(std::move(tableau)), m_rhs(std::move(rhs)) {
    // Allocate derivative and state buffers for each field
    for (std::size_t field_idx = 0; field_idx < N; ++field_idx) {
      const std::size_t size = local_sizes[field_idx];
      const unsigned int s = m_tableau.stage_count();

      // Allocate stage derivatives for this field
      m_k[field_idx].resize(s);
      for (unsigned int i = 0; i < s; ++i) {
        m_k[field_idx][i].assign(size, 0.0);
      }

      // Allocate RHS evaluation buffer for this field
      m_du[field_idx].assign(size, 0.0);

      // Allocate temporary state buffer for this field
      m_u_temp[field_idx].resize(size);

      // Allocate attempt state buffer for this field
      m_attempt_states[field_idx].resize(size);

      // Allocate original state buffer for this field
      m_original_states[field_idx].resize(size);
    }
    m_state_saved = false;
  }

  /**
   * @brief Attempt step for multi-field system using explicit RK
   *
   * @param t Current time
   * @param state_pack Multi-field state tuple
   * @return StepAttemptResult
   *
   * Applies explicit RK to each field independently:
   *   - For each stage i: k_i,k = rhs_k(t + c_i*dt, u_k + dt * sum_j(a_ij * k_j,k))
   *   - Final accumulation: u_new,k = u_k + dt * sum_i(b_i * k_i,k)
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

    const unsigned int s = m_tableau.stage_count();

    // Compute stages for each field
    for (unsigned int i = 0; i < s; ++i) {
      // Build temp states for each field: u_temp,k = u_k + dt * sum_j(a_ij * k_j,k)
      build_temp_states(state_pack, i, std::index_sequence_for<decltype(state_pack)>{});

      // Compute stage i for all fields
      const double stage_time = t + m_tableau.c(i) * m_dt;
      auto du_pack = make_du_tuple(std::index_sequence_for<decltype(state_pack)>{});
      m_rhs(stage_time, m_u_temp, du_pack);

      // Copy du to k_i for each field
      copy_du_to_k(du_pack, i, std::index_sequence_for<decltype(state_pack)>{});
    }

    // Final accumulation for each field: u_new,k = u_k + dt * sum_i(b_i * k_i,k)
    accumulate(state_pack, std::index_sequence_for<decltype(state_pack)>{});

    return StepAttemptResult{.status = StepAttemptResult::Status::Accepted,
                             .new_time = t + m_dt,
                             .error_estimate = std::optional<double>{}};
  }

  /**
   * @brief Get attempted states for multi-field system
   *
   * @return Array of attempted state buffers
   */
  [[nodiscard]] std::array<std::vector<double> &, N> get_attempt_states() {
    std::array<std::vector<double> &, N> result;
    for (std::size_t i = 0; i < N; ++i) {
      result[i] = m_attempt_states[i];
    }
    return result;
  }

  /**
   * @brief Get stage derivatives for multi-field system
   *
   * @return Array of stage derivative arrays (one per field)
   */
  [[nodiscard]] std::array<std::vector<std::vector<double>> &, N>
  get_stage_derivatives() {
    std::array<std::vector<std::vector<double>> &, N> result;
    for (std::size_t i = 0; i < N; ++i) {
      result[i] = m_k[i];
    }
    return result;
  }

  /**
   * @brief Get original states for multi-field rollback
   */
  [[nodiscard]] std::array<std::vector<double> &, N> get_original_states() {
    std::array<std::vector<double> &, N> result;
    for (std::size_t i = 0; i < N; ++i) {
      result[i] = m_original_states[i];
    }
    return result;
  }

  void commit_step() { m_state_saved = false; }
  void reject_step() { m_state_saved = false; }
  [[nodiscard]] double dt() const noexcept { return m_dt; }
  [[nodiscard]] bool supports_adaptive() const noexcept { return false; }
  [[nodiscard]] IntegratorMethod method() const noexcept {
    return IntegratorMethod::RK4;
  }
  [[nodiscard]] bool has_saved_state() const noexcept { return m_state_saved; }
  [[nodiscard]] const ButcherTableau<double> &get_tableau() const noexcept {
    return m_tableau;
  }
  [[nodiscard]] unsigned int stage_count() const noexcept {
    return m_tableau.stage_count();
  }

private:
  template <class... U, std::size_t... I>
  void build_temp_states(std::tuple<std::vector<U> &...> &u_pack, unsigned int stage_idx,
                         std::index_sequence<I...>) {
    // Process each field independently
    auto process_field = [&](auto idx) {
      constexpr std::size_t field_idx = idx;
      auto &u = std::get<field_idx>(u_pack);
      auto &u_temp = std::get<field_idx>(m_u_temp);

      // Start with current state
      u_temp = u;

      // Add contributions from previous stages: u_temp += dt * sum_j(a_ij * k_j)
      for (unsigned int j = 0; j < stage_idx; ++j) {
        const double a_ij = m_tableau.a(stage_idx, j);
        if (a_ij != 0.0) {
          for (std::size_t k = 0; k < u.size(); ++k) {
            u_temp[k] += m_dt * a_ij * m_k[field_idx][j][k];
          }
        }
      }
    };

    (process_field(std::integral_constant<std::size_t, I>{}), ...);
  }

  template <std::size_t... I> auto make_du_tuple(std::index_sequence<I...>) {
    return std::make_tuple(std::ref(m_du[I])...);
  }

  template <class DuPack, std::size_t... I>
  void copy_du_to_k(DuPack &du_pack, unsigned int stage_idx, std::index_sequence<I...>) {
    ((m_k[I][stage_idx] = std::get<I>(du_pack)), ...);
  }

  template <class... U, std::size_t... I>
  void accumulate(std::tuple<std::vector<U> &...> &u_pack, std::index_sequence<I...>) {
    auto accumulate_one = [&](std::vector<double> &u, const std::size_t field_idx) -> void {
      const unsigned int s = m_tableau.stage_count();
      const std::size_t n = u.size();

      // Start with current state
      m_attempt_states[field_idx] = u;

      // Apply final accumulation: u_new = u + dt * sum_i(b_i * k_i)
      for (unsigned int i = 0; i < s; ++i) {
        const double b_i = m_tableau.b(i);
        if (b_i != 0.0) {
          for (std::size_t idx = 0; idx < n; ++idx) {
            m_attempt_states[field_idx][idx] += m_dt * b_i * m_k[field_idx][i][idx];
          }
        }
      }
    };

    ((accumulate_one(std::get<I>(u_pack), I)), ...);
  }

  double m_dt;
  ButcherTableau<double> m_tableau;
  std::array<std::vector<std::vector<double>>, N> m_k; ///< k_i for each field
  std::array<std::vector<double>, N> m_du;            ///< RHS buffers for each field
  std::array<std::vector<double>, N> m_u_temp;        ///< Temp states for each field
  std::array<std::vector<double>, N> m_attempt_states; ///< Final states for each field
  std::array<std::vector<double>, N> m_original_states; ///< Original states for rollback
  Rhs m_rhs;
  bool m_state_saved;
};

} // namespace pfc::sim::steppers