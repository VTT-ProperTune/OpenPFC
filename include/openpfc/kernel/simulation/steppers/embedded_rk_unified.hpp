// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file embedded_rk_unified.hpp  
 * @brief Unified embedded Runge-Kutta stepper implementing M6 unified stepper protocol
 *
 * @details
 * `UnifiedEmbeddedRKStepper` implements the unified stepper protocol for
 * embedded explicit Runge-Kutta methods with adaptive error estimation.
 * These methods evaluate shared explicit stages once from an embedded
 * ButcherTableau and provide both high-order and low-order solutions
 * for error estimation.
 *
 * **Key Features:**
 * - Supports embedded Butcher tableaus (e.g., Dormand-Prince 4(5), Fehlberg 4(5))
 * - Provides adaptive error estimation via difference between high and low order solutions
 * - Adheres to M6 unified stepper protocol with attempt/commit semantics
 * - Works with general field-based state (not just std::vector<double>)
 * - Supports both single-field and multi-field variants
 * - Enables adaptive time step control via error estimates
 *
 * **Protocol Implementation:**
 * - `attempt_step(t, state)` computes both u_high and u_low, error = ||u_high - u_low||
 * - `commit_step()` applies the high-order solution to the accepted state  
 * - `reject_step()` rolls back to the checkpointed state
 * - Error estimate available for adaptive controllers
 *
 * **Embedded Methods:**
 * The stepper works with any embedded Butcher tableau that has:
 * - Standard weights `b` for the high-order method
 * - Embedded weights `b_hat` for the low-order method  
 * - Shared stage coefficients `a` and nodes `c`
 * 
 * Common embedded methods include:
 * - Dormand-Prince 4(5) - 5th order high, 4th order low
 * - Fehlberg 4(5) - alternative 4(5) pair
 * - Bogacki-Shampine 3(2) - 3rd order high, 2nd order low
 *
 * @see unified_stepper_protocol.hpp for M6 protocol requirements
 * @see embedded_rk.hpp for the original embedded RK implementation
 * @see butcher_tableau.hpp for embedded tableau factories
 * @see OPENPFC_REFACTORING_EXECUTION_PLAN.md M6 section for adaptive control
 */

#include <cmath>
#include <concepts>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <openpfc/kernel/simulation/steppers/butcher_tableau.hpp>
#include <openpfc/kernel/simulation/steppers/stage_protocol.hpp>
#include <openpfc/kernel/simulation/steppers/unified_stepper_protocol.hpp>

namespace pfc::sim::steppers {

/**
 * @brief Embedded RK method statistics from step attempt
 * 
 * Encapsulates performance and diagnostic information from an embedded
 * RK step, useful for adaptive control and performance analysis.
 */
struct EmbeddedRKStats {
  int stages_evaluated{0};            ///< Number of RK stages evaluated (should equal tableau.stage_count())
  double error_norm{0.0};             ///< Norm of the error estimate (||error||)
  double high_order_norm{0.0};        ///< Norm of the high-order solution
  double low_order_norm{0.0};         ///< Norm of the low-order solution
  bool error_estimate_valid{true};    ///< Whether error estimate is computationally valid
};

/**
 * @brief Unified embedded Runge-Kutta stepper for single fields
 *
 * Implements the M6 unified stepper protocol for embedded explicit RK
 * methods with adaptive error estimation capability.
 *
 * @tparam Rhs Callable satisfying StageFunction (rhs(t, u, du) computes du)
 *
 * Constructor: `UnifiedEmbeddedRKStepper(local_size, tableau, rhs)`
 *
 * @throws std::invalid_argument if !tableau.has_embedded()
 */
template <class Rhs>
  requires StageFunction<Rhs>
class UnifiedEmbeddedRKStepper {
public:
  /**
   * @brief Construct a unified embedded RK stepper
   * 
   * @param local_size Number of elements in the field
   * @param tableau Embedded Butcher tableau (must have has_embedded() == true)
   * @param rhs RHS callable for evaluating derivatives
   * 
   * @throws std::invalid_argument if tableau does not have embedded weights
   */
  UnifiedEmbeddedRKStepper(std::size_t local_size, ButcherTableau<double> tableau, Rhs rhs)
      : m_local_size(local_size), m_tableau(std::move(tableau)), m_rhs(std::move(rhs)),
        m_du(local_size, 0.0), m_u_temp(local_size, 0.0),
        m_u_high(local_size, 0.0), m_u_low(local_size, 0.0),
        m_error(local_size, 0.0), m_u_checkpoint(local_size, 0.0) {
    if (!m_tableau.has_embedded()) {
      throw std::invalid_argument(
          "UnifiedEmbeddedRKStepper requires an embedded ButcherTableau "
          "(has_embedded() == true; missing b_hat / embedded weights)");
    }
    
    // Allocate stage derivative buffers
    const unsigned int s = m_tableau.stage_count();
    m_k.resize(s);
    for (unsigned int i = 0; i < s; ++i) {
      m_k[i].assign(local_size, 0.0);
    }
  }

  /**
   * @brief Attempt one embedded RK step without mutating the accepted state
   * 
   * Evaluates all RK stages once, then forms both high-order and low-order
   * solutions. The error estimate is computed as:
   *     error = u_high - u_low
   * 
   * The state passed in is never modified; all work uses stepper-owned buffers.
   * 
   * @param t Current time
   * @param u_accepted Current accepted state (read-only)
   * @return StepAttemptResult with status, new time, and error estimate
   */
  [[nodiscard]] StepAttemptResult attempt_step(double t, 
                                                const std::vector<double> &u_accepted) {
    if (u_accepted.size() != m_local_size) {
      throw std::invalid_argument(
          "UnifiedEmbeddedRKStepper::attempt_step: u_accepted.size() (" +
          std::to_string(u_accepted.size()) + ") != local_size (" +
          std::to_string(m_local_size) + ")");
    }

    // Checkpoint the accepted state for potential rollback
    m_u_checkpoint = u_accepted;
    m_state_saved = true;
    
    const unsigned int s = m_tableau.stage_count();
    m_last_stats.stages_evaluated = static_cast<int>(s);
    
    // Evaluate all RK stages
    for (unsigned int i = 0; i < s; ++i) {
      // Build temporary state for this stage: u_temp = u + dt * sum(a_ij * k_j)
      m_u_temp = u_accepted;
      for (unsigned int j = 0; j < i; ++j) {
        const double a_ij = m_tableau.a(i, j);
        if (a_ij != 0.0) {
          for (std::size_t idx = 0; idx < m_local_size; ++idx) {
            m_u_temp[idx] += m_dt * a_ij * m_k[j][idx];
          }
        }
      }
      
      // Evaluate RHS at stage time: k_i = rhs(t + c_i * dt, u_temp)
      const double stage_time = t + m_tableau.c(i) * m_dt;
      m_rhs(stage_time, m_u_temp, m_du);
      m_k[i] = m_du;
    }
    
    // Form high-order solution: u_high = u + dt * sum(b_i * k_i)
    m_u_high = u_accepted;
    for (unsigned int i = 0; i < s; ++i) {
      const double b_i = m_tableau.b(i);
      if (b_i != 0.0) {
        for (std::size_t idx = 0; idx < m_local_size; ++idx) {
          m_u_high[idx] += m_dt * b_i * m_k[i][idx];
        }
      }
    }
    
    // Form low-order solution: u_low = u + dt * sum(b_hat_i * k_i)
    m_u_low = u_accepted;
    for (unsigned int i = 0; i < s; ++i) {
      const double b_hat_i = m_tableau.b_hat(i);
      if (b_hat_i != 0.0) {
        for (std::size_t idx = 0; idx < m_local_size; ++idx) {
          m_u_low[idx] += m_dt * b_hat_i * m_k[i][idx];
        }
      }
    }
    
    // Compute error estimate: error = u_high - u_low
    double error_norm = 0.0;
    double high_order_norm = 0.0;
    double low_order_norm = 0.0;
    
    for (std::size_t idx = 0; idx < m_local_size; ++idx) {
      m_error[idx] = m_u_high[idx] - m_u_low[idx];
      
      // Using L2 norm for error estimate
      const double err_val = m_error[idx];
      error_norm += err_val * err_val;
      
      const double high_val = m_u_high[idx];
      high_order_norm += high_val * high_val;
      
      const double low_val = m_u_low[idx];
      low_order_norm += low_val * low_val;
    }
    
    error_norm = std::sqrt(error_norm);
    high_order_norm = std::sqrt(high_order_norm);
    low_order_norm = std::sqrt(low_order_norm);
    
    // Store statistics
    m_last_stats.error_norm = error_norm;
    m_last_stats.high_order_norm = high_order_norm;
    m_last_stats.low_order_norm = low_order_norm;
    m_last_stats.error_estimate_valid = true;
    
    // Build result - always "Accepted" status since we successfully computed
    // The adaptive controller will decide whether to accept or reject based on error
    StepAttemptResult result;
    result.status = StepAttemptResult::Status::Accepted;
    result.new_time = t + m_dt;
    result.error_estimate = error_norm;
    
    m_last_attempt_result = result;
    return result;
  }

  /**
   * @brief Commit the attempted step using the high-order solution
   * 
   * Applies the high-order solution (u_high) to the accepted state.
   * Should only be called after a successful attempt_step where the
   * adaptive controller has deemed the error acceptable.
   * 
   * @param u_accepted The accepted state to update
   */
  void commit_step(std::vector<double> &u_accepted) {
    if (!m_state_saved) {
      throw std::logic_error(
          "UnifiedEmbeddedRKStepper::commit_step called without a prior attempt_step");
    }
    
    if (m_last_attempt_result.status != StepAttemptResult::Status::Accepted) {
      throw std::logic_error(
          "UnifiedEmbeddedRKStepper::commit_step called on a failed attempt");
    }
    
    if (u_accepted.size() != m_local_size) {
      throw std::invalid_argument(
          "UnifiedEmbeddedRKStepper::commit_step: u_accepted.size() (" +
          std::to_string(u_accepted.size()) + ") != local_size (" +
          std::to_string(m_local_size) + ")");
    }
    
    // Apply the high-order solution
    u_accepted = m_u_high;
  }

  /**
   * @brief Reject the attempted step and rollback to checkpointed state
   * 
   * Restores the accepted state to the checkpointed value from before
   * the attempt_step call. Used when adaptive controller finds error too large.
   * 
   * @param u_accepted The accepted state to rollback
   */
  void reject_step(std::vector<double> &u_accepted) {
    if (!m_state_saved) {
      throw std::logic_error(
          "UnifiedEmbeddedRKStepper::reject_step called without a prior attempt_step");
    }
    
    if (u_accepted.size() != m_local_size) {
      throw std::invalid_argument(
          "UnifiedEmbeddedRKStepper::reject_step: u_accepted.size() (" +
          std::to_string(u_accepted.size()) + ") != local_size (" +
          std::to_string(m_local_size) + ")");
    }
    
    // Rollback to the checkpointed state
    u_accepted = m_u_checkpoint;
    
    // Clear the saved state flag
    m_state_saved = false;
  }

  /**
   * @brief Get the current time step size
   */
  [[nodiscard]] double dt() const noexcept { return m_dt; }

  /**
   * @brief Set a new time step size
   */
  void set_dt(double new_dt) { m_dt = new_dt; }

  /**
   * @brief Check if this stepper supports adaptive error estimation
   * 
   * Embedded RK methods are designed for adaptive control, so this returns true.
   */
  [[nodiscard]] bool supports_adaptive() const noexcept { return true; }

  /**
   * @brief Get the integration method type
   */
  [[nodiscard]] IntegratorMethod method() const noexcept {
    return IntegratorMethod::EmbeddedRK;
  }

  /**
   * @brief Get the embedded Butcher tableau
   */
  [[nodiscard]] const ButcherTableau<double> &tableau() const noexcept {
    return m_tableau;
  }

  /**
   * @brief Get statistics from the last step attempt
   * 
   * @return Statistics including error norm, solution norms, stage count
   */
  [[nodiscard]] const EmbeddedRKStats &get_stats() const noexcept {
    return m_last_stats;
  }

  /**
   * @brief Access the high-order solution (for debugging/validation)
   */
  [[nodiscard]] const std::vector<double> &u_high() const noexcept {
    return m_u_high;
  }

  /**
   * @brief Access the low-order solution (for debugging/validation)
   */
  [[nodiscard]] const std::vector<double> &u_low() const noexcept {
    return m_u_low;
  }

  /**
   * @brief Access the error estimate (for debugging/validation)
   */
  [[nodiscard]] const std::vector<double> &error() const noexcept {
    return m_error;
  }

  /**
   * @brief Get field size
   */
  [[nodiscard]] std::size_t field_size() const noexcept { return m_local_size; }

private:
  std::size_t m_local_size{0};               ///< Number of field elements
  ButcherTableau<double> m_tableau;          ///< Embedded Butcher tableau
  Rhs m_rhs;                                 ///< RHS callable

  // Time step (can be changed between steps for adaptive control)
  double m_dt{0.01};                         ///< Current time step size

  // Working buffers
  std::vector<double> m_du;                  ///< Derivative buffer for RHS output
  std::vector<std::vector<double>> m_k;      ///< Stage derivative vectors
  std::vector<double> m_u_temp;              ///< Temporary state for stage evaluation
  std::vector<double> m_u_high;              ///< High-order solution candidate
  std::vector<double> m_u_low;               ///< Low-order solution  
  std::vector<double> m_error;               ///< Error estimate (u_high - u_low)
  std::vector<double> m_u_checkpoint;        ///< Checkpointed state for rollback

  // State tracking
  bool m_state_saved{false};                 ///< Whether state is checkpointed
  StepAttemptResult m_last_attempt_result;   ///< Result of last attempt
  EmbeddedRKStats m_last_stats;             ///< Statistics from last step
};

/**
 * @brief Unified embedded RK stepper for multiple fields (N-field packs)
 *
 * Extends the single-field unified embedded RK stepper to handle heterogeneous
 * multi-field packs. Each field gets its own set of working buffers, but
 * the same tableau and stage evaluation logic apply to all fields.
 *
 * @tparam Rhs Multi-field RHS callable satisfying StageFunction
 * @tparam N Number of fields in the pack
 */
template <class Rhs, std::size_t N>
class UnifiedMultiEmbeddedRKStepper {
public:
  using RHSType = Rhs;
  static constexpr std::size_t field_count = N;

  static_assert(N >= 1, "UnifiedMultiEmbeddedRKStepper requires N >= 1");

  /**
   * @brief Construct a unified multi-field embedded RK stepper
   * 
   * @param field_sizes Array of field sizes (one per field)
   * @param tableau Embedded Butcher tableau
   * @param rhs Multi-field RHS callable
   * 
   * @throws std::invalid_argument if tableau does not have embedded weights
   */
  UnifiedMultiEmbeddedRKStepper(const std::array<std::size_t, N> &field_sizes,
                               ButcherTableau<double> tableau, Rhs rhs)
      : m_tableau(std::move(tableau)), m_rhs(std::move(rhs)) {
    if (!m_tableau.has_embedded()) {
      throw std::invalid_argument(
          "UnifiedMultiEmbeddedRKStepper requires an embedded ButcherTableau");
    }
    
    for (std::size_t i = 0; i < N; ++i) {
      m_field_sizes[i] = field_sizes[i];
      
      // Allocate per-field buffers
      m_u_high[i].assign(field_sizes[i], 0.0);
      m_u_low[i].assign(field_sizes[i], 0.0);
      m_error[i].assign(field_sizes[i], 0.0);
      m_u_checkpoint[i].assign(field_sizes[i], 0.0);
    }
    
    // Allocate stage buffers
    const unsigned int s = m_tableau.stage_count();
    m_k.resize(s);
    for (unsigned int stage = 0; stage < s; ++stage) {
      for (std::size_t field = 0; field < N; ++field) {
        m_k[stage][field].assign(field_sizes[field], 0.0);
      }
    }
    
    // Allocate temporary buffers
    m_u_temp.resize(N);
    m_du.resize(N);
    for (std::size_t i = 0; i < N; ++i) {
      m_u_temp[i].assign(field_sizes[i], 0.0);
      m_du[i].assign(field_sizes[i], 0.0);
    }
  }

  /**
   * @brief Attempt one multi-field embedded RK step
   * 
   * @tparam U Field types (must be std::vector<double>)
   * @param t Current time
   * @param u_accepted Accepted states for all fields  
   * @return StepAttemptResult with status and error estimate
   */
  template <class... U>
  [[nodiscard]] StepAttemptResult attempt_step(double t,
                                                const std::vector<U> &...u_accepted) {
    static_assert(sizeof...(U) == N,
                  "UnifiedMultiEmbeddedRKStepper: field count must match N");
    static_assert((std::is_same_v<U, double> && ...),
                  "UnifiedMultiEmbeddedRKStepper requires std::vector<double>");

    // Checkpoint all fields
    save_checkpoint(u_accepted...);
    m_state_saved = true;
    
    const unsigned int s = m_tableau.stage_count();
    m_last_stats.stages_evaluated = static_cast<int>(s);
    
    // Evaluate all RK stages for all fields
    for (unsigned int i = 0; i < s; ++i) {
      // Build temporary states for this stage
      build_temp_states(i, u_accepted...);
      
      // Evaluate RHS at stage time
      const double stage_time = t + m_tableau.c(i) * m_dt;
      evaluate_rhs(stage_time);
      
      // Store stage derivatives
      store_stage_derivatives(i);
    }
    
    // Form high-order and low-order solutions
    form_solutions(u_accepted...);
    
    // Compute error estimate
    double error_norm = compute_error_norm();
    
    // Store statistics  
    m_last_stats.error_norm = error_norm;
    m_last_stats.error_estimate_valid = true;
    
    // Build result
    StepAttemptResult result;
    result.status = StepAttemptResult::Status::Accepted;
    result.new_time = t + m_dt;
    result.error_estimate = error_norm;
    
    m_last_attempt_result = result;
    return result;
  }

  /**
   * @brief Commit the attempted step for all fields
   * 
   * @tparam U Field types
   * @param u_accepted Accepted states to update
   */
  template <class... U>
  void commit_step(std::vector<U> &...u_accepted) {
    if (!m_state_saved) {
      throw std::logic_error(
          "UnifiedMultiEmbeddedRKStepper::commit_step called without a prior attempt_step");
    }
    
    if (m_last_attempt_result.status != StepAttemptResult::Status::Accepted) {
      throw std::logic_error(
          "UnifiedMultiEmbeddedRKStepper::commit_step called on a failed attempt");
    }
    
    // Apply high-order solutions to all fields
    std::size_t i = 0;
    ((u_accepted = m_u_high[i++]), ...);
  }

  /**
   * @brief Reject the attempted step and rollback all fields
   * 
   * @tparam U Field types
   * @param u_accepted Accepted states to rollback
   */
  template <class... U>
  void reject_step(std::vector<U> &...u_accepted) {
    if (!m_state_saved) {
      throw std::logic_error(
          "UnifiedMultiEmbeddedRKStepper::reject_step called without a prior attempt_step");
    }
    
    // Rollback all fields
    restore_checkpoint(u_accepted...);
    m_state_saved = false;
  }

  /**
   * @brief Get current time step
   */
  [[nodiscard]] double dt() const noexcept { return m_dt; }

  /**
   * @brief Set new time step
   */
  void set_dt(double new_dt) { m_dt = new_dt; }

  /**
   * @brief Check if adaptive control is supported
   */
  [[nodiscard]] bool supports_adaptive() const noexcept { return true; }

  /**
   * @brief Get integration method
   */
  [[nodiscard]] IntegratorMethod method() const noexcept {
    return IntegratorMethod::EmbeddedRK;
  }

  /**
   * @brief Get statistics from last step
   */
  [[nodiscard]] const EmbeddedRKStats &get_stats() const noexcept {
    return m_last_stats;
  }

  /**
   * @brief Get field sizes
   */
  [[nodiscard]] const std::array<std::size_t, N> &field_sizes() const noexcept {
    return m_field_sizes;
  }

private:
  template <class... U>
  void save_checkpoint(const std::vector<U> &...u_accepted) {
    std::size_t i = 0;
    ((m_u_checkpoint[i++] = u_accepted), ...);
  }

  template <class... U>
  void restore_checkpoint(std::vector<U> &...u_accepted) {
    std::size_t i = 0;
    ((u_accepted = m_u_checkpoint[i++]), ...);
  }

  template <class... U>
  void build_temp_states(unsigned int stage, const std::vector<U> &...u_accepted) {
    // Initialize temp states with accepted states
    std::size_t field_idx = 0;
    ((m_u_temp[field_idx++] = u_accepted), ...);
    
    // Add stage contributions from previous stages
    for (unsigned int j = 0; j < stage; ++j) {
      const double a_ij = m_tableau.a(stage, j);
      if (a_ij != 0.0) {
        field_idx = 0;
        auto add_contribution = [this, a_ij, j](std::vector<double> &temp, std::size_t idx) {
          for (std::size_t i = 0; i < temp.size(); ++i) {
            temp[i] += m_dt * a_ij * m_k[j][idx][i];
          }
        };
        ((add_contribution(m_u_temp[field_idx], field_idx++)), ...);
      }
    }
  }

  void evaluate_rhs(double stage_time) {
    // Build tuple for RHS evaluation - this assumes multi-field RHS expects tuple of refs
    auto u_tuple = make_tuple_from_array(m_u_temp);
    auto du_tuple = make_tuple_from_array(m_du);
    m_rhs(stage_time, u_tuple, du_tuple);
  }

  void store_stage_derivatives(unsigned int stage) {
    for (std::size_t field = 0; field < N; ++field) {
      m_k[stage][field] = m_du[field];
    }
  }

  template <class... U>
  void form_solutions(const std::vector<U> &...u_accepted) {
    // Initialize solutions with accepted states
    std::size_t field_idx = 0;
    ((m_u_high[field_idx] = u_accepted, m_u_low[field_idx++] = u_accepted), ...);
    
    const unsigned int s = m_tableau.stage_count();
    
    // Add stage contributions to high-order solution
    for (unsigned int i = 0; i < s; ++i) {
      const double b_i = m_tableau.b(i);
      if (b_i != 0.0) {
        field_idx = 0;
        auto add_high_contribution = [this, b_i, i](std::vector<double> &solution, std::size_t idx) {
          for (std::size_t j = 0; j < solution.size(); ++j) {
            solution[j] += m_dt * b_i * m_k[i][idx][j];
          }
        };
        ((add_high_contribution(m_u_high[field_idx], field_idx++)), ...);
      }
    }
    
    // Add stage contributions to low-order solution
    for (unsigned int i = 0; i < s; ++i) {
      const double b_hat_i = m_tableau.b_hat(i);
      if (b_hat_i != 0.0) {
        field_idx = 0;
        auto add_low_contribution = [this, b_hat_i, i](std::vector<double> &solution, std::size_t idx) {
          for (std::size_t j = 0; j < solution.size(); ++j) {
            solution[j] += m_dt * b_hat_i * m_k[i][idx][j];
          }
        };
        ((add_low_contribution(m_u_low[field_idx], field_idx++)), ...);
      }
    }
    
    // Compute error estimates
    field_idx = 0;
    auto compute_error = [this](std::size_t idx) {
      for (std::size_t i = 0; i < m_error[idx].size(); ++i) {
        m_error[idx][i] = m_u_high[idx][i] - m_u_low[idx][i];
      }
    };
    ((compute_error(field_idx++)), ...);
  }

  double compute_error_norm() {
    double error_norm = 0.0;
    for (std::size_t field = 0; field < N; ++field) {
      for (std::size_t i = 0; i < m_error[field].size(); ++i) {
        const double err_val = m_error[field][i];
        error_norm += err_val * err_val;
      }
    }
    return std::sqrt(error_norm);
  }

  template <std::size_t... I>
  auto make_tuple_from_array(std::array<std::vector<double>, N> &arr) {
    return std::tie(arr[I]...);
  }

  std::array<std::size_t, N> m_field_sizes;           ///< Sizes of each field
  ButcherTableau<double> m_tableau;                  ///< Embedded Butcher tableau
  Rhs m_rhs;                                         ///< Multi-field RHS callable

  // Time step
  double m_dt{0.01};                                 ///< Current time step size

  // Multi-field working buffers
  std::array<std::vector<double>, N> m_u_high;       ///< High-order solutions
  std::array<std::vector<double>, N> m_u_low;        ///< Low-order solutions
  std::array<std::vector<double>, N> m_error;        ///< Error estimates
  std::array<std::vector<double>, N> m_u_checkpoint; ///< Checkpointed states
  std::array<std::vector<double>, N> m_u_temp;       ///< Temporary states
  std::array<std::vector<double>, N> m_du;           ///< RHS outputs

  // Stage derivatives: m_k[stage][field]
  std::vector<std::array<std::vector<double>, N>> m_k; ///< Stage derivatives

  // State tracking
  bool m_state_saved{false};                         ///< Whether states are checkpointed
  StepAttemptResult m_last_attempt_result;           ///< Result of last attempt
  EmbeddedRKStats m_last_stats;                     ///< Statistics from last step
};

} // namespace pfc::sim::steppers