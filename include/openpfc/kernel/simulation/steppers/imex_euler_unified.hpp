// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file imex_euler_unified.hpp
 * @brief Unified IMEX Euler stepper implementing M6 unified stepper protocol
 *
 * @details
 * `UnifiedImexEulerStepper` implements the unified stepper protocol for
 * first-order IMEX (implicit-explicit) Euler time integration. This steppers
 * advances a field by one step using:
 *
 *     u_{n+1} = u_n + dt * (E(t_n, u_n) + I(t_{n+1}, u_{n+1}))
 *
 * where E is the explicit RHS and I is the implicit RHS. For linear-implicit
 * cases with linear operator L, this solves:
 *
 *     (I - dt * L) u_{n+1} = u_n + dt * E(t_n, u_n)
 *
 * **Key Features:**
 * - Adheres to M6 unified stepper protocol with attempt/commit semantics
 * - Supports both single-field and multi-field variants
 * - Integrates with the solver contract for implicit solves
 * - Provides rollback capability through state checkpoints
 * - Works with general field-based state (not just std::vector<double>)
 *
 * **Protocol Implementation:**
 * - `attempt_step(t, state)` computes candidate without mutating accepted state
 * - `commit_step()` applies the candidate to the accepted state
 * - `reject_step()` rolls back to the checkpointed state
 * - Error estimation provides solver convergence information
 *
 * @see unified_stepper_protocol.hpp for M6 protocol requirements
 * @see imex_euler.hpp for the original IMEX Euler implementation
 * @see solver_contract.hpp for SolveFunction interface
 */

#include <array>
#include <concepts>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/solver_contract.hpp>
#include <openpfc/kernel/simulation/steppers/stage_protocol.hpp>
#include <openpfc/kernel/simulation/steppers/unified_stepper_protocol.hpp>

namespace pfc::sim::steppers {

/**
 * @brief IMEX Euler solver statistics from implicit solve
 * 
 * Encapsulates information about the implicit solve performed during
 * the IMEX Euler step, useful for adaptive control and diagnostics.
 */
struct ImexSolverStats {
  bool converged{false};                          ///< Did the solver converge?
  int iterations{0};                              ///< Number of solver iterations
  double final_residual{0.0};                     ///< Final residual norm
  std::optional<std::string> failure_reason;      ///< Reason for failure (if any)
  double solve_time{0.0};                         ///< Time spent in solver (optional)
};

/**
 * @brief Unified IMEX Euler stepper for single fields
 *
 * Implements the M6 unified stepper protocol for first-order IMEX Euler
 * integration with attempt/commit semantics and rollback capability.
 *
 * @tparam ExplicitRhs Callable satisfying StageFunction for explicit RHS E(t, u)
 * @tparam Solver Callable modeling SolveFunction for implicit solve
 *
 * Constructor: `UnifiedImexEulerStepper(dt, field_size, E, solver, op_desc, opts)`
 */
template <class ExplicitRhs, class Solver>
  requires StageFunction<ExplicitRhs>
class UnifiedImexEulerStepper {
public:
  /**
   * @brief Construct a unified IMEX Euler stepper
   * 
   * @param dt Time step size
   * @param field_size Number of elements in the field
   * @param E Explicit RHS callable E(t, u, du) that computes du = E(t, u)
   * @param solver Implicit solver that solves (I - dt*L) u = rhs
   * @param op_desc Linear operator descriptor for the implicit part
   * @param opts Solver options (tolerances, max iterations, etc.)
   */
  UnifiedImexEulerStepper(double dt, std::size_t field_size, ExplicitRhs E,
                         Solver solver,
                         LinearOperatorDesc op_desc,
                         SolveOptions opts = {})
      : m_dt(dt), m_field_size(field_size), m_E(std::move(E)),
        m_solver(std::move(solver)), m_op_desc(std::move(op_desc)),
        m_opts(std::move(opts)),
        m_u_work(field_size, 0.0), m_e(field_size, 0.0),
        m_rhs_vec(field_size, 0.0), m_candidate(field_size, 0.0),
        m_u_checkpoint(field_size, 0.0) {}

  /**
   * @brief Attempt one IMEX Euler step without mutating the accepted state
   * 
   * Computes the candidate u_{n+1} without writing to the input state:
   * 1. Evaluate explicit RHS: e = E(t, u_accepted) 
   * 2. Form RHS: rhs = u_accepted + dt * e
   * 3. Solve implicit system: (I - dt*L) candidate = rhs
   * 4. Store solver statistics and candidate result
   * 
   * @param t Current time
   * @param u_accepted Current accepted state (read-only)
   * @return StepAttemptResult with status and error estimate (solver residual)
   */
  [[nodiscard]] StepAttemptResult attempt_step(double t, 
                                                const std::vector<double> &u_accepted) {
    if (u_accepted.size() != m_field_size) {
      throw std::invalid_argument(
          "UnifiedImexEulerStepper::attempt_step: u_accepted.size() (" +
          std::to_string(u_accepted.size()) + ") != field_size (" +
          std::to_string(m_field_size) + ")");
    }

    // Checkpoint the accepted state for potential rollback
    m_u_checkpoint = u_accepted;
    m_state_saved = true;
    
    // Copy accepted state to working buffer
    m_u_work = u_accepted;
    
    // Evaluate explicit RHS: e = E(t, u_work)
    double eval_time = t;
    m_E(eval_time, m_u_work, m_e);
    
    // Form RHS: rhs = u_accepted + dt * e
    for (std::size_t i = 0; i < m_field_size; ++i) {
      m_rhs_vec[i] = u_accepted[i] + m_dt * m_e[i];
    }
    
    // Setup stage context for implicit solve
    StageContext ctx;
    ctx.stage_time = t + m_dt;
    ctx.dt = m_dt;
    ctx.stage_number = 1;  // IMEX Euler has 1 stage
    ctx.total_stages = 1;
    ctx.method = IntegratorMethod::ImexEuler;
    ctx.evaluation_time = t + m_dt;
    
    // Solve implicit system: (I - dt*L) candidate = rhs
    auto rhs_bundle = std::tie(m_rhs_vec);
    auto candidate_bundle = std::tie(m_candidate);
    auto outcome = m_solver(m_op_desc, rhs_bundle, candidate_bundle, m_opts, ctx);
    
    // Store solver statistics
    m_last_solver_stats.converged = (outcome.status == ConvergenceStatus::converged);
    m_last_solver_stats.iterations = outcome.iteration_count;
    m_last_solver_stats.final_residual = outcome.final_residual_norm;
    m_last_solver_stats.failure_reason = outcome.failure_cause;
    
    // Determine step status based on solver convergence
    StepAttemptResult result;
    result.new_time = t + m_dt;
    
    if (m_last_solver_stats.converged) {
      result.status = StepAttemptResult::Status::Accepted;
      result.error_estimate = m_last_solver_stats.final_residual;
      
      // Copy solution from solver outcome if different from candidate buffer
      if constexpr (requires { outcome.solution; }) {
        if constexpr (requires { 
                         { std::get<0>(outcome.solution) } -> std::convertible_to<const std::vector<double>&>; 
                       }) {
          const std::vector<double> &solver_solution = std::get<0>(outcome.solution);
          if (&solver_solution != &m_candidate) {
            m_candidate = solver_solution;
          }
        }
      }
    } else {
      result.status = StepAttemptResult::Status::Failed;
      result.error_estimate = std::nullopt;  // No valid error estimate on failure
    }
    
    m_last_attempt_result = result;
    return result;
  }

  /**
   * @brief Commit the attempted step to the accepted state
   * 
   * Copies the candidate solution into the accepted state buffer.
   * Should only be called after a successful attempt_step that returned
   * StepAttemptResult::Status::Accepted.
   * 
   * @param u_accepted The accepted state to update
   */
  void commit_step(std::vector<double> &u_accepted) {
    if (!m_state_saved) {
      throw std::logic_error(
          "UnifiedImexEulerStepper::commit_step called without a prior attempt_step");
    }
    
    if (m_last_attempt_result.status != StepAttemptResult::Status::Accepted) {
      throw std::logic_error(
          "UnifiedImexEulerStepper::commit_step called on a failed attempt");
    }
    
    if (u_accepted.size() != m_field_size) {
      throw std::invalid_argument(
          "UnifiedImexEulerStepper::commit_step: u_accepted.size() (" +
          std::to_string(u_accepted.size()) + ") != field_size (" +
          std::to_string(m_field_size) + ")");
    }
    
    // Apply the candidate to the accepted state
    u_accepted = m_candidate;
  }

  /**
   * @brief Reject the attempted step and rollback to checkpointed state
   * 
   * Restores the accepted state to the checkpointed value from before
   * the attempt_step call. Invalidates the candidate and solver statistics.
   * 
   * @param u_accepted The accepted state to rollback
   */
  void reject_step(std::vector<double> &u_accepted) {
    if (!m_state_saved) {
      throw std::logic_error(
          "UnifiedImexEulerStepper::reject_step called without a prior attempt_step");
    }
    
    if (u_accepted.size() != m_field_size) {
      throw std::invalid_argument(
          "UnifiedImexEulerStepper::reject_step: u_accepted.size() (" +
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
   * IMEX Euler itself doesn't provide embedded error estimation, but
   * the solver residual can be used as an error indicator. Returns true
   * to indicate that some error information is available.
   */
  [[nodiscard]] bool supports_adaptive() const noexcept { return true; }

  /**
   * @brief Get the integration method type
   */
  [[nodiscard]] IntegratorMethod method() const noexcept {
    return IntegratorMethod::ImexEuler;
  }

  /**
   * @brief Get solver statistics from the last attempt
   * 
   * @return Solver statistics including convergence status, iterations, residual
   */
  [[nodiscard]] const ImexSolverStats &get_solver_stats() const noexcept {
    return m_last_solver_stats;
  }

  /**
   * @brief Get field size
   */
  [[nodiscard]] std::size_t field_size() const noexcept { return m_field_size; }

  /**
   * @brief Get explicit RHS callable
   */
  [[nodiscard]] const ExplicitRhs &get_explicit_rhs() const noexcept {
    return m_E;
  }

  /**
   * @brief Get implicit solver
   */
  [[nodiscard]] const Solver &get_solver() const noexcept { return m_solver; }

private:
  double m_dt{0.0};                          ///< Time step size
  std::size_t m_field_size{0};               ///< Number of field elements
  ExplicitRhs m_E;                           ///< Explicit RHS callable
  Solver m_solver;                           ///< Implicit solver
  LinearOperatorDesc m_op_desc;              ///< Linear operator descriptor
  SolveOptions m_opts;                       ///< Solver options

  // Working buffers
  std::vector<double> m_u_work;              ///< Working copy of u for explicit eval
  std::vector<double> m_e;                   ///< Explicit RHS result E(t, u)
  std::vector<double> m_rhs_vec;             ///< RHS for implicit solve: u + dt*e
  std::vector<double> m_candidate;           ///< Candidate solution u_{n+1}
  std::vector<double> m_u_checkpoint;        ///< Checkpointed state for rollback

  // State tracking
  bool m_state_saved{false};                 ///< Whether state is checkpointed
  StepAttemptResult m_last_attempt_result;   ///< Result of last attempt
  ImexSolverStats m_last_solver_stats;       ///< Statistics from last solve
};

/**
 * @brief Unified IMEX Euler stepper for multiple fields (N-field packs)
 *
 * Extends the single-field unified IMEX Euler stepper to handle heterogeneous
 * multi-field packs using structure-of-arrays layout.
 *
 * @tparam ExplicitRhs Multi-field explicit RHS callable
 * @tparam Solver Injected SolveFunction-compatible solver
 * @tparam N Number of fields in the pack
 */
template <class ExplicitRhs, class Solver, std::size_t N>
class UnifiedMultiImexEulerStepper {
public:
  using ExplicitRhsType = ExplicitRhs;
  static constexpr std::size_t field_count = N;

  static_assert(N >= 1, "UnifiedMultiImexEulerStepper requires N >= 1");

  /**
   * @brief Construct a unified multi-field IMEX Euler stepper
   * 
   * @param dt Time step size
   * @param field_sizes Array of field sizes (one per field)
   * @param E Multi-field explicit RHS callable
   * @param solver Implicit solver for the coupled system
   * @param op_desc Linear operator descriptor
   * @param opts Solver options
   */
  UnifiedMultiImexEulerStepper(double dt, 
                              const std::array<std::size_t, N> &field_sizes,
                              ExplicitRhs E,
                              Solver solver,
                              LinearOperatorDesc op_desc,
                              SolveOptions opts = {})
      : m_dt(dt), m_E(std::move(E)), m_solver(std::move(solver)),
        m_op_desc(std::move(op_desc)), m_opts(std::move(opts)) {
    for (std::size_t i = 0; i < N; ++i) {
      m_field_sizes[i] = field_sizes[i];
      m_u_work[i].assign(field_sizes[i], 0.0);
      m_e[i].assign(field_sizes[i], 0.0);
      m_rhs_vec[i].assign(field_sizes[i], 0.0);
      m_candidate[i].assign(field_sizes[i], 0.0);
      m_u_checkpoint[i].assign(field_sizes[i], 0.0);
    }
  }

  /**
   * @brief Attempt one multi-field IMEX Euler step
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
                  "UnifiedMultiImexEulerStepper: field count must match N");
    static_assert((std::is_same_v<U, double> && ...),
                  "UnifiedMultiImexEulerStepper requires std::vector<double>");

    // Checkpoint all fields for potential rollback
    save_checkpoint(u_accepted...);
    m_state_saved = true;
    
    // Copy accepted states to working buffers
    copy_accepted_to_work(u_accepted...);
    
    // Evaluate explicit RHS for all fields
    auto u_pack = make_work_tuple();
    auto e_pack = make_e_tuple();
    m_E(t, u_pack, e_pack);
    
    // Form RHS for all fields: rhs = u + dt * e
    form_rhs(u_accepted...);
    
    // Setup stage context
    StageContext ctx;
    ctx.stage_time = t + m_dt;
    ctx.dt = m_dt;
    ctx.stage_number = 1;
    ctx.total_stages = 1;
    ctx.method = IntegratorMethod::ImexEuler;
    ctx.evaluation_time = t + m_dt;
    
    // Solve implicit system
    auto rhs_bundle = make_rhs_bundle();
    auto candidate_bundle = make_candidate_bundle();
    auto outcome = m_solver(m_op_desc, rhs_bundle, candidate_bundle, m_opts, ctx);
    
    // Store solver statistics
    m_last_solver_stats.converged = (outcome.status == ConvergenceStatus::converged);
    m_last_solver_stats.iterations = outcome.iteration_count;
    m_last_solver_stats.final_residual = outcome.final_residual_norm;
    m_last_solver_stats.failure_reason = outcome.failure_cause;
    
    // Determine step status
    StepAttemptResult result;
    result.new_time = t + m_dt;
    
    if (m_last_solver_stats.converged) {
      result.status = StepAttemptResult::Status::Accepted;
      result.error_estimate = m_last_solver_stats.final_residual;
      
      // Copy solution from solver outcome if needed
      ingest_solution(outcome.solution);
    } else {
      result.status = StepAttemptResult::Status::Failed;
      result.error_estimate = std::nullopt;
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
          "UnifiedMultiImexEulerStepper::commit_step called without a prior attempt_step");
    }
    
    if (m_last_attempt_result.status != StepAttemptResult::Status::Accepted) {
      throw std::logic_error(
          "UnifiedMultiImexEulerStepper::commit_step called on a failed attempt");
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
          "UnifiedMultiImexEulerStepper::reject_step called without a prior attempt_step");
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
  [[nodiscard]] bool supports_adaptive() const noexcept { return true; }

  /**
   * @brief Get integration method
   */
  [[nodiscard]] IntegratorMethod method() const noexcept {
    return IntegratorMethod::ImexEuler;
  }

  /**
   * @brief Get solver statistics
   */
  [[nodiscard]] const ImexSolverStats &get_solver_stats() const noexcept {
    return m_last_solver_stats;
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
  void copy_accepted_to_work(const std::vector<U> &...u_accepted) {
    std::size_t i = 0;
    ((m_u_work[i++] = u_accepted), ...);
  }

  template <class... U>
  void copy_candidate_to_accepted(std::vector<U> &...u_accepted) {
    std::size_t i = 0;
    ((u_accepted = m_candidate[i++]), ...);
  }

  auto make_work_tuple() {
    return make_tuple_from_array<0>(m_u_work);
  }

  auto make_e_tuple() {
    return make_tuple_from_array<0>(m_e);
  }

  template <std::size_t... I>
  auto make_tuple_from_array(std::array<std::vector<double>, N> &arr) {
    return std::tie(arr[I]...);
  }

  template <class... U>
  void form_rhs(const std::vector<U> &...u_accepted) {
    auto form_one = [this](std::vector<double> &rhs, 
                          const std::vector<double> &u,
                          const std::vector<double> &e,
                          std::size_t idx) {
      for (std::size_t i = 0; i < u.size(); ++i) {
        rhs[i] = u[i] + m_dt * e[i];
      }
    };
    
    std::size_t idx = 0;
    ((form_one(m_rhs_vec[idx], u_accepted, m_e[idx], idx++)), ...);
  }

  auto make_rhs_bundle() {
    return make_tuple_from_array<0>(m_rhs_vec);
  }

  auto make_candidate_bundle() {
    return make_tuple_from_array<0>(m_candidate);
  }

  template <class Solution>
  void ingest_solution(Solution &&solution) {
    if constexpr (requires { 
                     { std::get<0>(solution) } -> std::convertible_to<const std::vector<double>&>; 
                   }) {
      std::size_t i = 0;
      auto copy_one = [this](std::vector<double> &dest, 
                            const std::vector<double> &src,
                            std::size_t idx) {
        if (&src != &dest) {
          dest = src;
        }
      };
      
      ((copy_one(m_candidate[i], std::get<i>(solution), i++)), ...);
    }
    // else: assume solver already wrote into candidate buffers in place
  }

  double m_dt{0.0};                          ///< Time step size
  std::array<std::size_t, N> m_field_sizes;  ///< Sizes of each field
  ExplicitRhs m_E;                           ///< Explicit RHS callable
  Solver m_solver;                           ///< Implicit solver
  LinearOperatorDesc m_op_desc;              ///< Linear operator descriptor
  SolveOptions m_opts;                       ///< Solver options

  // Working buffers for each field
  std::array<std::vector<double>, N> m_u_work;    ///< Working copies
  std::array<std::vector<double>, N> m_e;         ///< Explicit RHS results
  std::array<std::vector<double>, N> m_rhs_vec;   ///< RHS for implicit solve
  std::array<std::vector<double>, N> m_candidate; ///< Candidate solutions
  std::array<std::vector<double>, N> m_u_checkpoint; ///< Checkpointed states

  // State tracking
  bool m_state_saved{false};                 ///< Whether states are checkpointed
  StepAttemptResult m_last_attempt_result;   ///< Result of last attempt
  ImexSolverStats m_last_solver_stats;       ///< Statistics from last solve
};

} // namespace pfc::sim::steppers