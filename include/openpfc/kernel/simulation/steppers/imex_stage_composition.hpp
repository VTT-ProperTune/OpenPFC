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
 * **StageContext:** `pfc::sim::StageContext` is an alias of
 * `pfc::integrator::StageContext`. The composer writes `ctx.time`.
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
#include <openpfc/kernel/simulation/steppers/step_attempt.hpp>

namespace pfc::sim::steppers {

/**
 * @brief Explicit operator evaluator: fill `du` from read-only accepted state.
 *
 * Stronger isolation than historical `StageFunction`: the concept requires a
 * const accepted buffer so evaluators cannot write through `u`.
 */
template <class F>
concept ExplicitOperatorEval = requires(F f, double t, const std::vector<double> &u,
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
   * 2. Set `ctx.time = t` (composer owns this write for the proof path).
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
   * @param ctx Stage context; `time` is set by this call.
   * @return `StepAttemptResult`. Solve extras are `last_solve_*` accessors.
   *
   * @throws std::invalid_argument if `u_accepted.size() != local_size`.
   */
  [[nodiscard]] StepAttemptResult attempt(double t, double dt,
                                          const std::vector<double> &u_accepted,
                                          const LinearOperatorDesc &op_desc,
                                          const SolveOptions &options,
                                          StageContext &ctx) {
    if (u_accepted.size() != m_local_size) {
      throw std::invalid_argument("ImexEulerComposer::attempt: u_accepted.size() (" +
                                  std::to_string(u_accepted.size()) +
                                  ") != local_size (" +
                                  std::to_string(m_local_size) + ")");
    }

    ctx.time = t;

    m_explicit_eval(t, u_accepted, m_f_explicit);

    for (std::size_t i = 0; i < m_local_size; ++i) {
      m_rhs[i] = u_accepted[i] + dt * m_f_explicit[i];
    }

    const auto outcome = m_solver(op_desc, m_rhs, m_candidate, options, ctx);
    m_last_solve_status = outcome.status;
    m_last_solve_iteration_count = outcome.iteration_count;
    m_last_solve_final_residual_norm = outcome.final_residual_norm;
    m_last_solve_failure_cause = outcome.failure_cause;

    if (outcome.status != ConvergenceStatus::converged) {
      return StepAttemptResult(t, dt, t, /*success=*/false, m_candidate);
    }

    return StepAttemptResult(t, dt, t + dt, /*success=*/true, m_candidate);
  }

  [[nodiscard]] std::optional<ConvergenceStatus> last_solve_status() const noexcept {
    return m_last_solve_status;
  }
  [[nodiscard]] int last_solve_iteration_count() const noexcept {
    return m_last_solve_iteration_count;
  }
  [[nodiscard]] double last_solve_final_residual_norm() const noexcept {
    return m_last_solve_final_residual_norm;
  }
  [[nodiscard]] const std::optional<std::string> &
  last_solve_failure_cause() const noexcept {
    return m_last_solve_failure_cause;
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

  [[nodiscard]] const std::vector<double> &candidate() const { return m_candidate; }

  [[nodiscard]] std::size_t local_size() const { return m_local_size; }

private:
  std::size_t m_local_size;
  std::vector<double> m_f_explicit;
  std::vector<double> m_rhs;
  std::vector<double> m_candidate;
  ExplicitEval m_explicit_eval;
  Solver m_solver;
  std::optional<ConvergenceStatus> m_last_solve_status;
  int m_last_solve_iteration_count{0};
  double m_last_solve_final_residual_norm{0.0};
  std::optional<std::string> m_last_solve_failure_cause;
};

} // namespace pfc::sim::steppers
