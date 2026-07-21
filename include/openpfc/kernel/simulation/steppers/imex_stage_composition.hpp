// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file imex_stage_composition.hpp
 * @brief IMEX stage-composition seam: explicit eval then implicit SolveFunction.
 *
 * @details
 * Provides a kernel-visible composition API that sequences
 * stage-context preparation → explicit operator evaluation into stage
 * storage → implicit `pfc::sim::SolveFunction` solve into an isolated
 * candidate buffer, without mutating accepted state until the driver calls
 * `apply_candidate`.
 *
 * There is **no** virtual `ImexIntegrator` base class. Method authors compose
 * callables (`ExplicitOperatorEval` + `SolveFunction`) through
 * `ImexEulerComposer` (CPU IMEX-Euler-shaped proof path). Full product IMEX
 * Euler (#168), spectral/Krylov solver bodies, and CUDA/HIP backends are out
 * of scope for this seam.
 *
 * **StageContext namespace:** always use `pfc::sim::StageContext` from
 * `solver_contract.hpp`. Do not confuse it with `pfc::integrator::StageContext`.
 *
 * @see openpfc/kernel/simulation/solver_contract.hpp
 * @see openpfc/kernel/simulation/steppers/embedded_rk.hpp
 */

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <openpfc/kernel/simulation/solver_contract.hpp>

namespace pfc::sim::steppers {

/**
 * @brief Evidence returned by one IMEX stage-composition attempt.
 *
 * `candidate` binds to method-owned storage and remains valid until the next
 * `attempt` call or composer destruction. On failure, `t1` is unspecified
 * (left equal to `t0`); inspect `success` / `solve_status` before reading
 * candidate contents. Accepted input buffers are never written by `attempt`.
 */
struct ImexStepAttemptResult {
  double t0{};
  double dt{};
  double t1{}; ///< `t0 + dt` on success; equal to `t0` on failure (unspecified)
  bool success{false};
  ConvergenceStatus solve_status{ConvergenceStatus::unknown_failure};
  int solve_iterations{0};
  double final_residual_norm{0.0};
  std::optional<std::string> failure_cause;
  const std::vector<double> &candidate;

  ImexStepAttemptResult(double t0_in, double dt_in, double t1_in, bool success_in,
                        ConvergenceStatus solve_status_in,
                        int solve_iterations_in, double final_residual_norm_in,
                        std::optional<std::string> failure_cause_in,
                        const std::vector<double> &candidate_in)
      : t0(t0_in), dt(dt_in), t1(t1_in), success(success_in),
        solve_status(solve_status_in), solve_iterations(solve_iterations_in),
        final_residual_norm(final_residual_norm_in),
        failure_cause(std::move(failure_cause_in)), candidate(candidate_in) {}
};

/**
 * @brief Explicit operator evaluator: fill `du` from read-only accepted state.
 *
 * Stronger isolation than historical `StageFunction`: the concept requires a
 * const accepted buffer so evaluators cannot write through `u`.
 */
template <class F>
concept ExplicitOperatorEval =
    requires(F f, double t, const std::vector<double> &u,
             std::vector<double> &du) { f(t, u, du); };

/**
 * @brief CPU IMEX-Euler-shaped stage composer (proof path).
 *
 * Sequences one explicit evaluation and one implicit solve into an isolated
 * candidate. Models the shared IMEX composition seam without a virtual base.
 *
 * For each successful attempt the implicit problem is the first-order form
 * @f$(I - \Delta t L_I) u^\ast = u + \Delta t f_E(t,u)@f$, where @f$f_E@f$ is
 * produced by `ExplicitEval` and @f$L_I@f$ is identified by `op_desc`. The
 * composer builds the RHS vector and hands it to `Solver`; it does not own
 * a spectral/Krylov implementation.
 *
 * @tparam ExplicitEval Models `ExplicitOperatorEval`
 * @tparam Solver Models `pfc::sim::SolveFunction` for
 *         `std::vector<double>` RHS and target
 */
template <class ExplicitEval, class Solver>
  requires ExplicitOperatorEval<ExplicitEval> &&
           SolveFunction<Solver, std::vector<double>, std::vector<double>>
class ImexEulerComposer {
public:
  /**
   * @brief Construct a composer owning stage/candidate buffers of `local_size`.
   */
  ImexEulerComposer(std::size_t local_size, ExplicitEval eval, Solver solver)
      : m_local_size(local_size), m_f_explicit(local_size, 0.0),
        m_rhs(local_size, 0.0), m_candidate(local_size, 0.0),
        m_explicit_eval(std::move(eval)), m_solver(std::move(solver)) {}

  /**
   * @brief Attempt one IMEX-Euler-shaped step without mutating accepted state.
   *
   * Sequence (fixed order):
   * 1. Validate `u_accepted.size() == local_size`.
   * 2. Set `ctx.evaluation_time = t` (composer owns this write for the proof
   *    path).
   * 3. Explicit stage: `m_explicit_eval(t, u_accepted, m_f_explicit)`.
   * 4. Build RHS: `m_rhs[i] = u_accepted[i] + dt * m_f_explicit[i]`.
   * 5. Implicit solve into `m_candidate` via `m_solver`.
   * 6. On non-converged outcome: return `success=false`; leave accepted
   *    bitwise unchanged; `m_candidate` contents are unspecified (last
   *    failed attempt / prior value).
   * 7. On converged: return `success=true`, `t1 = t + dt`, metrics from
   *    outcome, `candidate` bound to `m_candidate`.
   *
   * @param t Current accepted time.
   * @param dt Proposed step size.
   * @param u_accepted Accepted state (read-only; never written).
   * @param op_desc Implicit linear operator descriptor for the solve.
   * @param options Solver stopping criteria.
   * @param ctx `pfc::sim::StageContext` (solver_contract); evaluation_time
   *            is set by this call.
   * @return Attempt evidence with a view into method-owned candidate storage.
   *
   * @throws std::invalid_argument if `u_accepted.size() != local_size`.
   */
  [[nodiscard]] ImexStepAttemptResult
  attempt(double t, double dt, const std::vector<double> &u_accepted,
          const LinearOperatorDesc &op_desc, const SolveOptions &options,
          StageContext &ctx) {
    if (u_accepted.size() != m_local_size) {
      throw std::invalid_argument(
          "ImexEulerComposer::attempt: u_accepted.size() (" +
          std::to_string(u_accepted.size()) + ") != local_size (" +
          std::to_string(m_local_size) + ")");
    }

    ctx.evaluation_time = t;

    m_explicit_eval(t, u_accepted, m_f_explicit);

    for (std::size_t i = 0; i < m_local_size; ++i) {
      m_rhs[i] = u_accepted[i] + dt * m_f_explicit[i];
    }

    const auto outcome = m_solver(op_desc, m_rhs, m_candidate, options, ctx);

    if (outcome.status != ConvergenceStatus::converged) {
      return ImexStepAttemptResult(
          t, dt, t, /*success=*/false, outcome.status, outcome.iteration_count,
          outcome.final_residual_norm, outcome.failure_cause, m_candidate);
    }

    return ImexStepAttemptResult(
        t, dt, t + dt, /*success=*/true, outcome.status, outcome.iteration_count,
        outcome.final_residual_norm, outcome.failure_cause, m_candidate);
  }

  /**
   * @brief Commit helper: copy the last successful candidate into accepted
   *        storage.
   *
   * Call only after a successful `attempt`. This is the sole write path from
   * the proof composer into accepted state.
   *
   * @throws std::invalid_argument if `u_inout.size() != local_size`.
   */
  void apply_candidate(std::vector<double> &u_inout) const {
    if (u_inout.size() != m_local_size) {
      throw std::invalid_argument(
          "ImexEulerComposer::apply_candidate: u_inout.size() (" +
          std::to_string(u_inout.size()) + ") != local_size (" +
          std::to_string(m_local_size) + ")");
    }
    u_inout = m_candidate;
  }

  [[nodiscard]] const std::vector<double> &candidate() const {
    return m_candidate;
  }

  [[nodiscard]] std::size_t local_size() const { return m_local_size; }

private:
  std::size_t m_local_size;
  std::vector<double> m_f_explicit;
  std::vector<double> m_rhs;
  std::vector<double> m_candidate;
  ExplicitEval m_explicit_eval;
  Solver m_solver;
};

} // namespace pfc::sim::steppers
