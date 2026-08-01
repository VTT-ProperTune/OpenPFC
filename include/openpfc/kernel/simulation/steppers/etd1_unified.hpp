// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file etd1_unified.hpp
 * @brief Unified ETD1 stepper implementing M6 unified stepper protocol
 *
 * @details
 * `UnifiedEtd1Stepper` implements the unified stepper protocol for
 * first-order exponential time differencing (ETD1) time integration.
 * This method is designed for stiff linear problems in spectral space
 * and is widely used in phase-field simulations (e.g., tungsten, aluminum).
 *
 * **Key Features:**
 * - Handles linear stiff systems via exact exponential propagation
 * - Efficient for spectral methods with diagonal linear operators
 * - Adheres to M6 unified stepper protocol with attempt/commit semantics
 * - Supports real and complex fields (spectral space typically uses complex)
 * - Works with both single-field and multi-field variants
 * - Enables coefficient ownership flexibility (caller-lent or method-owned)
 *
 * **ETD1 Formula:**
 * For a spectral ODE system: du/dt = L*u + N(u,t) where L is a linear operator
 * and N is the nonlinear term, ETD1 computes:
 *
 *     u_{n+1} = exp(dt*L) * u_n + phi_1(dt*L) * N(u_n, t_n) * dt
 *
 * where phi_1(z) = (exp(z)-1)/z with phi_1(0) = 1.
 *
 * **Coefficient Handling:**
 * - **Caller-lent spans**: `set_coefficients(exp_Ldt_span, phi1_L_span)` - views
 *   must remain valid until next call or destruction
 * - **Method-owned copies**: `set_coefficients_owned(exp_Ldt, phi1_L)` - copies
 *   into internal vectors so source may be dropped
 * - **SpectralExpCoefficientCache overload**: extracts coefficients from cache
 *
 * **Protocol Implementation:**
 * - `attempt_step(t, state)` computes candidate using spectral ETD formula
 * - `commit_step()` applies the candidate to the accepted state
 * - `reject_step()` rolls back to the checkpointed state
 * - Works with complex spectral fields for production applications
 *
 * @see unified_stepper_protocol.hpp for M6 protocol requirements
 * @see etd1.hpp for the original ETD1 implementation
 * @see spectral_exp_coefficients.hpp for coefficient computation
 * @see OPENPFC_REFACTORING_EXECUTION_PLAN.md M6 section for ETD skeleton
 */

#include <array>
#include <complex>
#include <concepts>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/kernel/simulation/steppers/stage_protocol.hpp>
#include <openpfc/kernel/simulation/steppers/unified_stepper_protocol.hpp>

namespace pfc::sim::steppers {

/**
 * @brief ETD1 step statistics from attempt
 * 
 * Encapsulates information about the ETD1 step computation,
 * useful for diagnostics and adaptive control.
 */
struct Etd1Stats {
  bool coefficients_valid{true};       ///< Whether ETD coefficients are valid
  bool nonlinear_eval_success{true};   ///< Whether nonlinear evaluation succeeded
  bool finite_values{true};            ///< Whether results contain no NaN/Inf
  double solution_norm{0.0};           ///< Norm of the candidate solution
  double nonlinear_norm{0.0};          ///< Norm of the nonlinear term
  std::string failure_reason;          ///< Reason for failure (if any)
};

/**
 * @brief Helper to create ETD1 success result
 */
[[nodiscard]] inline Etd1Stats make_etd1_success(double solution_norm, 
                                                 double nonlinear_norm) {
  Etd1Stats stats;
  stats.coefficients_valid = true;
  stats.nonlinear_eval_success = true;
  stats.finite_values = true;
  stats.solution_norm = solution_norm;
  stats.nonlinear_norm = nonlinear_norm;
  return stats;
}

/**
 * @brief Helper to create ETD1 failure result
 */
[[nodiscard]] inline Etd1Stats make_etd1_failure(std::string reason) {
  Etd1Stats stats;
  stats.coefficients_valid = false;
  stats.nonlinear_eval_success = false;
  stats.finite_values = false;
  stats.failure_reason = std::move(reason);
  return stats;
}

/**
 * @brief Unified ETD1 stepper for single fields (real or complex)
 *
 * Implements the M6 unified stepper protocol for first-order exponential
 * time differencing with attempt/commit semantics and rollback capability.
 *
 * @tparam Rhs Callable satisfying StageFunction (rhs(t, u, du) computes nonlinear term)
 * @tparam T Field value type (double for real fields, complex<double> for spectral)
 *
 * Constructor: `UnifiedEtd1Stepper(dt, field_size, rhs)`
 *
 * Typical usage pattern:
 * 1. Set coefficients: `stepper.set_coefficients(exp_Ldt_span, phi1_L_span)`
 * 2. Attempt step: `result = stepper.attempt_step(t, u_accepted)`
 * 3. Check result and decide accept/reject (adaptive controller)
 * 4. Commit or rollback: `stepper.commit_step(u_accepted)` or `stepper.reject_step(u_accepted)`
 */
template <class Rhs, typename T = double>
  requires StageFunction<Rhs>
class UnifiedEtd1Stepper {
public:
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
                "UnifiedEtd1Stepper requires T to be double or complex<double>");

  /**
   * @brief Construct a unified ETD1 stepper
   * 
   * @param dt Time step size
   * @param field_size Number of elements in the field
   * @param rhs Nonlinear RHS callable N(t, u, du) that computes du = N(t, u)
   */
  UnifiedEtd1Stepper(double dt, std::size_t field_size, Rhs rhs)
      : m_dt(dt), m_field_size(field_size), m_rhs(std::move(rhs)),
        m_du(field_size, T{}), m_candidate(field_size, T{}),
        m_u_scratch(field_size, T{}), m_u_checkpoint(field_size, T{}) {
    // Allocate owned coefficient buffers (will be overwritten when set_coefficients is called)
    m_exp_Ldt_owned.assign(field_size, T{});
    m_phi1_L_owned.assign(field_size, T{});
  }

  /**
   * @brief Bind caller-lent coefficient spans
   * 
   * @param exp_Ldt_span Span of exp(dt*L) coefficients
   * @param phi1_L_span Span of phi_1(dt*L) coefficients
   * 
   * @throws std::invalid_argument if span sizes don't match field_size
   * @note Views must remain valid until next set_coefficients call or destruction
   */
  void set_coefficients(std::span<const T> exp_Ldt_span, 
                       std::span<const T> phi1_L_span) {
    if (exp_Ldt_span.size() != m_field_size || phi1_L_span.size() != m_field_size) {
      throw std::invalid_argument(
          "UnifiedEtd1Stepper::set_coefficients: coefficient span sizes (" +
          std::to_string(exp_Ldt_span.size()) + ", " + 
          std::to_string(phi1_L_span.size()) + ") != field_size (" +
          std::to_string(m_field_size) + ")");
    }
    
    m_exp_Ldt = exp_Ldt_span;
    m_phi1_L = phi1_L_span;
    m_owns_coefficients = false;
  }

  /**
   * @brief Copy coefficients into method-owned storage
   * 
   * @param exp_Ldt Vector of exp(dt*L) coefficients
   * @param phi1_L Vector of phi_1(dt*L) coefficients
   * 
   * @throws std::invalid_argument if vector sizes don't match field_size
   * @note Source vectors can be dropped after this call
   */
  void set_coefficients_owned(const std::vector<T> &exp_Ldt, 
                             const std::vector<T> &phi1_L) {
    if (exp_Ldt.size() != m_field_size || phi1_L.size() != m_field_size) {
      throw std::invalid_argument(
          "UnifiedEtd1Stepper::set_coefficients_owned: coefficient vector sizes (" +
          std::to_string(exp_Ldt.size()) + ", " + 
          std::to_string(phi1_L.size()) + ") != field_size (" +
          std::to_string(m_field_size) + ")");
    }
    
    m_exp_Ldt_owned = exp_Ldt;
    m_phi1_L_owned = phi1_L;
    
    // Point spans to owned storage
    m_exp_Ldt = m_exp_Ldt_owned;
    m_phi1_L = m_phi1_L_owned;
    m_owns_coefficients = true;
  }

  /**
   * @brief Extract coefficients from SpectralExpCoefficientCache
   * 
   * @param cache SpectralExpCoefficientCache containing exp_Ldt and phi1_L
   * 
   * @throws std::invalid_argument if cache field size doesn't match
   */
  void set_coefficients(const integrator::SpectralExpCoefficientCache &cache) {
    if (cache.size() != m_field_size) {
      throw std::invalid_argument(
          "UnifiedEtd1Stepper::set_coefficients: cache size (" +
          std::to_string(cache.size()) + ") != field_size (" +
          std::to_string(m_field_size) + ")");
    }
    
    // Extract coefficients from cache and store in owned buffers
    m_exp_Ldt_owned.clear();
    m_phi1_L_owned.clear();
    m_exp_Ldt_owned.reserve(m_field_size);
    m_phi1_L_owned.reserve(m_field_size);
    
    for (std::size_t i = 0; i < m_field_size; ++i) {
      m_exp_Ldt_owned.push_back(static_cast<T>(cache.exp_Ldt(i)));
      m_phi1_L_owned.push_back(static_cast<T>(cache.phi1_L(i)));
    }
    
    // Point spans to owned storage
    m_exp_Ldt = m_exp_Ldt_owned;
    m_phi1_L = m_phi1_L_owned;
    m_owns_coefficients = true;
  }

  /**
   * @brief Attempt one ETD1 step without mutating the accepted state
   * 
   * Computes the candidate using the ETD1 formula:
   *     u_{n+1} = exp(dt*L) * u_n + phi_1(dt*L) * N(u_n, t_n) * dt
   * 
   * The state passed in is never modified; all work uses stepper-owned buffers.
   * 
   * @param t Current time
   * @param u_accepted Current accepted state (read-only)
   * @return StepAttemptResult with status and new time
   * 
   * @throws std::logic_error if coefficients have not been set
   */
  [[nodiscard]] StepAttemptResult attempt_step(double t, 
                                                const std::vector<T> &u_accepted) {
    if (u_accepted.size() != m_field_size) {
      throw std::invalid_argument(
          "UnifiedEtd1Stepper::attempt_step: u_accepted.size() (" +
          std::to_string(u_accepted.size()) + ") != field_size (" +
          std::to_string(m_field_size) + ")");
    }

    if (!m_exp_Ldt.data() || !m_phi1_L.data()) {
      throw std::logic_error(
          "UnifiedEtd1Stepper::attempt_step called before set_coefficients");
    }

    // Checkpoint the accepted state for potential rollback
    m_u_checkpoint = u_accepted;
    m_state_saved = true;
    
    // Copy accepted state to scratch buffer for nonlinear evaluation
    m_u_scratch = u_accepted;
    
    // Evaluate nonlinear term: du = N(t, u_scratch)
    try {
      m_rhs(t, m_u_scratch, m_du);
      m_last_stats.nonlinear_eval_success = true;
    } catch (const std::exception &e) {
      m_last_stats = make_etd1_failure(std::string("Nonlinear RHS evaluation failed: ") + e.what());
      
      StepAttemptResult result;
      result.status = StepAttemptResult::Status::Failed;
      result.new_time = t + m_dt;
      result.error_estimate = std::nullopt;
      
      m_last_attempt_result = result;
      return result;
    }
    
    // Apply ETD1 update: u_{n+1} = exp(dt*L) * u_n + phi_1(dt*L) * N * dt
    double solution_norm = 0.0;
    double nonlinear_norm = 0.0;
    bool finite_values = true;
    
    for (std::size_t i = 0; i < m_field_size; ++i) {
      // Compute ETD1 update
      const T exp_term = m_exp_Ldt[i] * u_accepted[i];
      const T nonlinear_term = m_phi1_L[i] * m_du[i] * m_dt;
      m_candidate[i] = exp_term + nonlinear_term;
      
      // Check for finite values
      if constexpr (std::is_same_v<T, std::complex<double>>) {
        if (!std::isfinite(m_candidate[i].real()) || !std::isfinite(m_candidate[i].imag())) {
          finite_values = false;
        }
        solution_norm += std::norm(m_candidate[i]);
        nonlinear_norm += std::norm(m_du[i]);
      } else {
        if (!std::isfinite(m_candidate[i])) {
          finite_values = false;
        }
        solution_norm += m_candidate[i] * m_candidate[i];
        nonlinear_norm += m_du[i] * m_du[i];
      }
    }
    
    solution_norm = std::sqrt(solution_norm);
    nonlinear_norm = std::sqrt(nonlinear_norm);
    
    // Update statistics
    m_last_stats.coefficients_valid = true;
    m_last_stats.finite_values = finite_values;
    m_last_stats.solution_norm = solution_norm;
    m_last_stats.nonlinear_norm = nonlinear_norm;
    
    // Determine step status
    StepAttemptResult result;
    result.new_time = t + m_dt;
    
    if (finite_values) {
      result.status = StepAttemptResult::Status::Accepted;
      result.error_estimate = std::nullopt;  // ETD1 doesn't provide embedded error estimate
    } else {
      result.status = StepAttemptResult::Status::Failed;
      result.error_estimate = std::nullopt;
      m_last_stats.failure_reason = "Non-finite values detected in candidate solution";
    }
    
    m_last_attempt_result = result;
    return result;
  }

  /**
   * @brief Commit the attempted step to the accepted state
   * 
   * Copies the ETD1 candidate solution into the accepted state buffer.
   * Should only be called after a successful attempt_step that returned
   * StepAttemptResult::Status::Accepted.
   * 
   * @param u_accepted The accepted state to update
   */
  void commit_step(std::vector<T> &u_accepted) {
    if (!m_state_saved) {
      throw std::logic_error(
          "UnifiedEtd1Stepper::commit_step called without a prior attempt_step");
    }
    
    if (m_last_attempt_result.status != StepAttemptResult::Status::Accepted) {
      throw std::logic_error(
          "UnifiedEtd1Stepper::commit_step called on a failed attempt");
    }
    
    if (u_accepted.size() != m_field_size) {
      throw std::invalid_argument(
          "UnifiedEtd1Stepper::commit_step: u_accepted.size() (" +
          std::to_string(u_accepted.size()) + ") != field_size (" +
          std::to_string(m_field_size) + ")");
    }
    
    // Apply the ETD1 candidate to the accepted state
    u_accepted = m_candidate;
  }

  /**
   * @brief Reject the attempted step and rollback to checkpointed state
   * 
   * Restores the accepted state to the checkpointed value from before
   * the attempt_step call.
   * 
   * @param u_accepted The accepted state to rollback
   */
  void reject_step(std::vector<T> &u_accepted) {
    if (!m_state_saved) {
      throw std::logic_error(
          "UnifiedEtd1Stepper::reject_step called without a prior attempt_step");
    }
    
    if (u_accepted.size() != m_field_size) {
      throw std::invalid_argument(
          "UnifiedEtd1Stepper::reject_step: u_accepted.size() (" +
          std::to_string(u_accepted.size()) + ") != field_size (" +
          std::to_string(m_field_size) + ")");
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
   * ETD1 itself doesn't provide embedded error estimation, so this returns false.
   * Adaptive control for ETD1 would require external error estimation mechanisms.
   */
  [[nodiscard]] bool supports_adaptive() const noexcept { return false; }

  /**
   * @brief Get the integration method type
   */
  [[nodiscard]] IntegratorMethod method() const noexcept {
    return IntegratorMethod::ETD1;
  }

  /**
   * @brief Get statistics from the last step attempt
   */
  [[nodiscard]] const Etd1Stats &get_stats() const noexcept {
    return m_last_stats;
  }

  /**
   * @brief Get field size
   */
  [[nodiscard]] std::size_t field_size() const noexcept { return m_field_size; }

  /**
   * @brief Get nonlinear RHS callable
   */
  [[nodiscard]] const Rhs &get_rhs() const noexcept { return m_rhs; }

  /**
   * @brief Check if coefficients are set
   */
  [[nodiscard]] bool has_coefficients() const noexcept {
    return m_exp_Ldt.data() != nullptr && m_phi1_L.data() != nullptr;
  }

  /**
   * @brief Check if coefficients are method-owned
   */
  [[nodiscard]] bool owns_coefficients() const noexcept {
    return m_owns_coefficients;
  }

private:
  double m_dt{0.0};                          ///< Time step size
  std::size_t m_field_size{0};               ///< Number of field elements
  Rhs m_rhs;                                 ///< Nonlinear RHS callable

  // Coefficient storage and views
  std::vector<T> m_exp_Ldt_owned;            ///< Owned exp(dt*L) coefficients
  std::vector<T> m_phi1_L_owned;             ///< Owned phi_1(dt*L) coefficients
  std::span<const T> m_exp_Ldt;              ///< View of exp(dt*L) coefficients
  std::span<const T> m_phi1_L;               ///< View of phi_1(dt*L) coefficients
  bool m_owns_coefficients{false};           ///< Whether we own the coefficients

  // Working buffers
  std::vector<T> m_du;                       ///< Nonlinear term N(t, u)
  std::vector<T> m_candidate;                ///< ETD1 candidate solution u_{n+1}
  std::vector<T> m_u_scratch;                ///< Scratch copy for nonlinear eval
  std::vector<T> m_u_checkpoint;             ///< Checkpointed state for rollback

  // State tracking
  bool m_state_saved{false};                 ///< Whether state is checkpointed
  StepAttemptResult m_last_attempt_result;   ///< Result of last attempt
  Etd1Stats m_last_stats;                    ///< Statistics from last step
};

/**
 * @brief Unified ETD1 stepper for multiple fields
 *
 * Extends the single-field unified ETD1 stepper to handle heterogeneous
 * multi-field packs. Each field gets its own coefficients and working buffers.
 *
 * @tparam Rhs Multi-field RHS callable
 * @tparam N Number of fields in the pack
 * @tparam T Field value type (default: double)
 */
template <class Rhs, std::size_t N, typename T = double>
class UnifiedMultiEtd1Stepper {
public:
  static_assert(N >= 1, "UnifiedMultiEtd1Stepper requires N >= 1");
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
                "UnifiedMultiEtd1Stepper requires T to be double or complex<double>");

  /**
   * @brief Construct a unified multi-field ETD1 stepper
   * 
   * @param dt Time step size
   * @param field_sizes Array of field sizes (one per field)
   * @param rhs Multi-field nonlinear RHS callable
   */
  UnifiedMultiEtd1Stepper(double dt, 
                         const std::array<std::size_t, N> &field_sizes,
                         Rhs rhs)
      : m_dt(dt), m_rhs(std::move(rhs)) {
    for (std::size_t i = 0; i < N; ++i) {
      m_field_sizes[i] = field_sizes[i];
      
      // Allocate per-field working buffers
      m_du[i].assign(field_sizes[i], T{});
      m_candidate[i].assign(field_sizes[i], T{});
      m_u_scratch[i].assign(field_sizes[i], T{});
      m_u_checkpoint[i].assign(field_sizes[i], T{});
      
      // Allocate per-field coefficient buffers
      m_exp_Ldt_owned[i].assign(field_sizes[i], T{});
      m_phi1_L_owned[i].assign(field_sizes[i], T{});
    }
  }

  /**
   * @brief Set coefficients for all fields
   * 
   * @param exp_Ldt_spans Array of coefficient spans (one per field)
   * @param phi1_L_spans Array of coefficient spans (one per field)
   */
  void set_coefficients(const std::array<std::span<const T>, N> &exp_Ldt_spans,
                       const std::array<std::span<const T>, N> &phi1_L_spans) {
    for (std::size_t i = 0; i < N; ++i) {
      if (exp_Ldt_spans[i].size() != m_field_sizes[i] || 
          phi1_L_spans[i].size() != m_field_sizes[i]) {
        throw std::invalid_argument(
            "UnifiedMultiEtd1Stepper::set_coefficients: coefficient span size mismatch");
      }
      m_exp_Ldt[i] = exp_Ldt_spans[i];
      m_phi1_L[i] = phi1_L_spans[i];
    }
    m_owns_coefficients = false;
  }

  /**
   * @brief Attempt one multi-field ETD1 step
   * 
   * @tparam U Field types (must be std::vector<T>)
   * @param t Current time
   * @param u_accepted Accepted states for all fields
   * @return StepAttemptResult with status and new time
   */
  template <class... U>
  [[nodiscard]] StepAttemptResult attempt_step(double t,
                                                const std::vector<U> &...u_accepted) {
    static_assert(sizeof...(U) == N,
                  "UnifiedMultiEtd1Stepper: field count must match N");
    static_assert((std::is_same_v<U, T> && ...),
                  "UnifiedMultiEtd1Stepper requires matching field types");

    // Checkpoint all fields
    save_checkpoint(u_accepted...);
    m_state_saved = true;
    
    // Copy accepted states to scratch buffers
    copy_accepted_to_scratch(u_accepted...);
    
    // Evaluate nonlinear terms for all fields
    try {
      auto u_scratch_tuple = make_tuple_from_array(m_u_scratch);
      auto du_tuple = make_tuple_from_array(m_du);
      m_rhs(t, u_scratch_tuple, du_tuple);
      m_last_stats.nonlinear_eval_success = true;
    } catch (const std::exception &e) {
      m_last_stats = make_etd1_failure(std::string("Nonlinear RHS evaluation failed: ") + e.what());
      
      StepAttemptResult result;
      result.status = StepAttemptResult::Status::Failed;
      result.new_time = t + m_dt;
      result.error_estimate = std::nullopt;
      
      m_last_attempt_result = result;
      return result;
    }
    
    // Apply ETD1 update to all fields
    apply_etd1_update(u_accepted...);
    
    // Build result
    StepAttemptResult result;
    result.status = m_last_stats.finite_values ? 
                    StepAttemptResult::Status::Accepted : 
                    StepAttemptResult::Status::Failed;
    result.new_time = t + m_dt;
    result.error_estimate = std::nullopt;
    
    if (!m_last_stats.finite_values) {
      m_last_stats.failure_reason = "Non-finite values detected in candidate solution";
    }
    
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
          "UnifiedMultiEtd1Stepper::commit_step called without a prior attempt_step");
    }
    
    if (m_last_attempt_result.status != StepAttemptResult::Status::Accepted) {
      throw std::logic_error(
          "UnifiedMultiEtd1Stepper::commit_step called on a failed attempt");
    }
    
    copy_candidate_to_accepted(u_accepted...);
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
          "UnifiedMultiEtd1Stepper::reject_step called without a prior attempt_step");
    }
    
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
  [[nodiscard]] bool supports_adaptive() const noexcept { return false; }

  /**
   * @brief Get integration method
   */
  [[nodiscard]] IntegratorMethod method() const noexcept {
    return IntegratorMethod::ETD1;
  }

  /**
   * @brief Get statistics from last step
   */
  [[nodiscard]] const Etd1Stats &get_stats() const noexcept {
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
  void copy_accepted_to_scratch(const std::vector<U> &...u_accepted) {
    std::size_t i = 0;
    ((m_u_scratch[i++] = u_accepted), ...);
  }

  template <class... U>
  void copy_candidate_to_accepted(std::vector<U> &...u_accepted) {
    std::size_t i = 0;
    ((u_accepted = m_candidate[i++]), ...);
  }

  template <class... U>
  void apply_etd1_update(const std::vector<U> &...u_accepted) {
    m_last_stats.finite_values = true;
    m_last_stats.solution_norm = 0.0;
    m_last_stats.nonlinear_norm = 0.0;
    
    auto apply_one_field = [this](std::size_t field_idx, const std::vector<U> &u_accepted) {
      double field_solution_norm = 0.0;
      double field_nonlinear_norm = 0.0;
      bool field_finite = true;
      
      for (std::size_t i = 0; i < m_field_sizes[field_idx]; ++i) {
        // ETD1 update: u_{n+1} = exp(dt*L) * u_n + phi_1(dt*L) * N * dt
        const T exp_term = m_exp_Ldt[field_idx][i] * u_accepted[i];
        const T nonlinear_term = m_phi1_L[field_idx][i] * m_du[field_idx][i] * m_dt;
        m_candidate[field_idx][i] = exp_term + nonlinear_term;
        
        // Check for finite values
        if constexpr (std::is_same_v<T, std::complex<double>>) {
          if (!std::isfinite(m_candidate[field_idx][i].real()) || 
              !std::isfinite(m_candidate[field_idx][i].imag())) {
            field_finite = false;
          }
          field_solution_norm += std::norm(m_candidate[field_idx][i]);
          field_nonlinear_norm += std::norm(m_du[field_idx][i]);
        } else {
          if (!std::isfinite(m_candidate[field_idx][i])) {
            field_finite = false;
          }
          field_solution_norm += m_candidate[field_idx][i] * m_candidate[field_idx][i];
          field_nonlinear_norm += m_du[field_idx][i] * m_du[field_idx][i];
        }
      }
      
      m_last_stats.finite_values = m_last_stats.finite_values && field_finite;
      m_last_stats.solution_norm += std::sqrt(field_solution_norm);
      m_last_stats.nonlinear_norm += std::sqrt(field_nonlinear_norm);
    };
    
    std::size_t field_idx = 0;
    ((apply_one_field(field_idx++, u_accepted)), ...);
  }

  template <std::size_t... I>
  auto make_tuple_from_array(std::array<std::vector<T>, N> &arr) {
    return std::tie(arr[I]...);
  }

  double m_dt{0.0};                                 ///< Time step size
  std::array<std::size_t, N> m_field_sizes;         ///< Sizes of each field
  Rhs m_rhs;                                        ///< Multi-field RHS callable

  // Per-field coefficient storage and views
  std::array<std::vector<T>, N> m_exp_Ldt_owned;    ///< Owned exp(dt*L) coefficients
  std::array<std::vector<T>, N> m_phi1_L_owned;     ///< Owned phi_1(dt*L) coefficients
  std::array<std::span<const T>, N> m_exp_Ldt;      ///< Views of exp(dt*L) coefficients
  std::array<std::span<const T>, N> m_phi1_L;       ///< Views of phi_1(dt*L) coefficients
  bool m_owns_coefficients{false};                  ///< Whether we own the coefficients

  // Per-field working buffers
  std::array<std::vector<T>, N> m_du;               ///< Nonlinear terms
  std::array<std::vector<T>, N> m_candidate;        ///< ETD1 candidates
  std::array<std::vector<T>, N> m_u_scratch;        ///< Scratch copies
  std::array<std::vector<T>, N> m_u_checkpoint;     ///< Checkpointed states

  // State tracking
  bool m_state_saved{false};                        ///< Whether states are checkpointed
  StepAttemptResult m_last_attempt_result;          ///< Result of last attempt
  Etd1Stats m_last_stats;                           ///< Statistics from last step
};

} // namespace pfc::sim::steppers