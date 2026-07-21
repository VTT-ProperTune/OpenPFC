// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file imex_euler.hpp
 * @brief First-order IMEX Euler stepper with explicit–implicit split.
 *
 * @details
 * Advances a field (or multi-field pack) by one first-order IMEX Euler step
 *
 *     u_{n+1} = u_n + dt * ( E(t_n, u_n) + I(t_{n+1}, u_{n+1}) )
 *
 * with the linear-implicit specialization
 *
 *     (I - dt * L) u_{n+1} = u_n + dt * E(t_n, u_n)
 *
 * when the implicit part is a linear operator L. Each attempt performs exactly
 * one explicit `StageFunction` / `MultiStageFunction` evaluation at
 * `(t_n, u_n)` and one injected `SolveFunction` solve for the implicit stage
 * at `t_{n+1}`. The stepper never hard-wires HeFFTe or spectral types; the
 * caller supplies `LinearOperatorDesc` plus a `SolveFunction`-compatible
 * solver (test double, diagonal helper, or production solver).
 *
 * ## Accepted-state isolation
 *
 * `attempt` never mutates the caller's accepted buffers. Explicit evaluation
 * runs on an internal working copy. The candidate `u_{n+1}` lives in
 * stepper-owned storage. On solve failure, `ImexStepAttempt::success` is
 * false, solve evidence is reported, and `commit` is a no-op — accepted state
 * is unchanged. `candidate()` refers to stepper-owned storage and is
 * meaningful only after a successful attempt.
 *
 * @see stage_protocol.hpp for StageFunction / MultiStageFunction
 * @see openpfc/kernel/simulation/solver_contract.hpp for SolveFunction,
 *      LinearOperatorDesc, SolveOutcome
 * @see euler.hpp for the purely explicit forward-Euler sibling
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <openpfc/kernel/simulation/solver_contract.hpp>
#include <openpfc/kernel/simulation/steppers/stage_protocol.hpp>

namespace pfc::sim::steppers {

/**
 * @brief Result of one IMEX Euler step attempt (not an in-place time advance).
 *
 * On failure, `success` is false, solve evidence is populated from
 * `SolveOutcome`, and the candidate must not be committed.
 */
struct ImexStepAttempt {
  bool success{false};
  double t_new{0.0}; ///< Meaningful only when `success` is true.
  std::optional<pfc::sim::ConvergenceStatus> solve_status;
  int solve_iteration_count{0};
  double solve_final_residual_norm{0.0};
  std::optional<std::string> solve_failure_cause;
};

namespace detail {

template <class Solution>
void ingest_single_field_solution(std::vector<double> &dest, Solution &&solution) {
  using S = std::remove_cvref_t<Solution>;
  if constexpr (std::is_same_v<S, std::vector<double>>) {
    dest = std::forward<Solution>(solution);
  } else if constexpr (requires {
                         {
                           std::get<0>(solution)
                         } -> std::convertible_to<const std::vector<double> &>;
                       }) {
    const std::vector<double> &src = std::get<0>(solution);
    if (std::addressof(src) != std::addressof(dest)) {
      dest = src;
    }
  }
  // else: assume the solver already wrote into `dest` in place
}

template <class Solution, std::size_t N, std::size_t... I>
void ingest_multi_field_solution_impl(std::array<std::vector<double>, N> &dest,
                                      Solution &&solution,
                                      std::index_sequence<I...>) {
  auto copy_one = [](std::vector<double> &d, auto &&src) {
    using Src = std::remove_cvref_t<decltype(src)>;
    if constexpr (std::is_same_v<Src, std::vector<double>>) {
      if (std::addressof(src) != std::addressof(d)) {
        d = src;
      }
    }
  };
  if constexpr (requires { (static_cast<void>(std::get<I>(solution)), ...); }) {
    (copy_one(dest[I], std::get<I>(solution)), ...);
  } else {
    (void)dest;
    (void)solution;
  }
}

template <class Solution, std::size_t N>
void ingest_multi_field_solution(std::array<std::vector<double>, N> &dest,
                                 Solution &&solution) {
  ingest_multi_field_solution_impl(
      dest, std::forward<Solution>(solution), std::make_index_sequence<N>{});
}

template <class Outcome>
ImexStepAttempt make_attempt_from_outcome(bool success, double t_new,
                                          const Outcome &outcome) {
  ImexStepAttempt attempt;
  attempt.success = success;
  attempt.t_new = t_new;
  attempt.solve_status = outcome.status;
  attempt.solve_iteration_count = outcome.iteration_count;
  attempt.solve_final_residual_norm = outcome.final_residual_norm;
  attempt.solve_failure_cause = outcome.failure_cause;
  return attempt;
}

} // namespace detail

/**
 * @brief Element-wise diagonal IMEX solver: `target[i] = rhs[i] / diag[i]`.
 *
 * Reads the diagonal of `(I - dt * L)` from
 * `LinearOperatorDesc::operator_context` as `std::vector<double>`. Returns
 * `ill_conditioned` if any `|diag[i]| < diag_eps`. Single-field bundles
 * (`std::tie(vector)` / one-element tuples) only — no HeFFTe coupling.
 *
 * @param diag_eps Absolute threshold below which a diagonal entry is treated
 *                 as singular (default `1e-14`).
 */
[[nodiscard]] inline auto make_diagonal_imex_solver(double diag_eps = 1e-14) {
  return [diag_eps](const pfc::sim::LinearOperatorDesc &op_desc,
                    const auto &rhs_bundle, auto &target_bundle,
                    const pfc::sim::SolveOptions & /*opts*/,
                    const pfc::sim::StageContext & /*ctx*/)
             -> pfc::sim::SolveOutcome<std::decay_t<decltype(target_bundle)>> {
    using TargetType = std::decay_t<decltype(target_bundle)>;
    if (!std::holds_alternative<std::vector<double>>(op_desc.operator_context)) {
      return pfc::sim::SolveOutcome<TargetType>{
          target_bundle, pfc::sim::ConvergenceStatus::ill_conditioned, 0, 0.0,
          std::string(
              "make_diagonal_imex_solver requires vector operator_context")};
    }
    const auto &diag = std::get<std::vector<double>>(op_desc.operator_context);
    const auto &rhs_vec = std::get<0>(rhs_bundle);
    auto &target_vec = std::get<0>(target_bundle);
    if (diag.size() != rhs_vec.size() || diag.size() != target_vec.size()) {
      return pfc::sim::SolveOutcome<TargetType>{
          target_bundle, pfc::sim::ConvergenceStatus::ill_conditioned, 0, 0.0,
          std::string("diagonal size mismatch vs rhs/target")};
    }
    for (std::size_t i = 0; i < diag.size(); ++i) {
      if (std::abs(diag[i]) < diag_eps) {
        return pfc::sim::SolveOutcome<TargetType>{
            target_bundle, pfc::sim::ConvergenceStatus::ill_conditioned, 0, 0.0,
            std::string("near-zero diagonal entry")};
      }
      target_vec[i] = rhs_vec[i] / diag[i];
    }
    return pfc::sim::SolveOutcome<TargetType>{
        target_bundle, pfc::sim::ConvergenceStatus::converged, 1, 0.0,
        std::nullopt};
  };
}

/**
 * @brief First-order IMEX Euler stepper for a single field.
 *
 * @tparam ExplicitRhs Callable satisfying `StageFunction` (explicit half E).
 * @tparam Solver      Callable modeling `SolveFunction` for the implicit half.
 *
 * Constructor:
 * `ImexEulerStepper(dt, local_size, E, solver, op_desc, opts = {})`.
 */
template <class ExplicitRhs, class Solver>
  requires StageFunction<ExplicitRhs>
class ImexEulerStepper {
public:
  ImexEulerStepper(double dt, std::size_t local_size, ExplicitRhs E,
                   Solver solver, pfc::sim::LinearOperatorDesc op_desc,
                   pfc::sim::SolveOptions opts = {})
      : m_dt(dt), m_E(std::move(E)), m_solver(std::move(solver)),
        m_op_desc(std::move(op_desc)), m_opts(std::move(opts)),
        m_u_work(local_size, 0.0), m_e(local_size, 0.0),
        m_rhs_vec(local_size, 0.0), m_candidate(local_size, 0.0),
        m_u_checkpoint(local_size, 0.0) {}

  /**
   * @brief Attempt one IMEX Euler step without mutating `u_accepted`.
   *
   * Evaluates `E(t, u_work)`, forms `rhs = u_accepted + dt * E`, then solves
   * `(I - dt*L) candidate = rhs` via the injected solver with
   * `ctx.evaluation_time = t + dt`.
   */
  [[nodiscard]] ImexStepAttempt attempt(double t,
                                        const std::vector<double> &u_accepted,
                                        pfc::sim::StageContext &ctx) {
    m_last_success = false;
    m_u_work = u_accepted;
    m_E(t, m_u_work, m_e);
    const std::size_t n = u_accepted.size();
    for (std::size_t i = 0; i < n; ++i) {
      m_rhs_vec[i] = u_accepted[i] + m_dt * m_e[i];
    }
    ctx.evaluation_time = t + m_dt;
    auto rhs_bundle = std::tie(m_rhs_vec);
    auto candidate_bundle = std::tie(m_candidate);
    auto outcome =
        m_solver(m_op_desc, rhs_bundle, candidate_bundle, m_opts, ctx);
    if (outcome.status == pfc::sim::ConvergenceStatus::converged) {
      detail::ingest_single_field_solution(m_candidate, outcome.solution);
      m_last_success = true;
      return detail::make_attempt_from_outcome(true, t + m_dt, outcome);
    }
    return detail::make_attempt_from_outcome(false, t + m_dt, outcome);
  }

  /**
   * @brief Copy the candidate into `u_accepted` only if the last attempt
   *        succeeded.
   * @return true if committed; false if accepted state was left unchanged.
   */
  [[nodiscard]] bool commit(std::vector<double> &u_accepted) const {
    if (!m_last_success) {
      return false;
    }
    u_accepted = m_candidate;
    return true;
  }

  /**
   * @brief View into the stepper-owned candidate buffer.
   * @pre The last `attempt` returned `success == true`.
   */
  [[nodiscard]] std::span<const double> candidate() const noexcept {
    return m_candidate;
  }

  [[nodiscard]] double dt() const noexcept { return m_dt; }

  void save_state(const std::vector<double> &u) { m_u_checkpoint = u; }

  void restore_state(std::vector<double> &u) { u = m_u_checkpoint; }

  [[nodiscard]] bool can_rollback() const noexcept { return true; }

private:
  double m_dt{0.0};
  ExplicitRhs m_E;
  Solver m_solver;
  pfc::sim::LinearOperatorDesc m_op_desc;
  pfc::sim::SolveOptions m_opts;
  std::vector<double> m_u_work;
  std::vector<double> m_e;
  std::vector<double> m_rhs_vec;
  std::vector<double> m_candidate;
  std::vector<double> m_u_checkpoint;
  bool m_last_success{false};
};

/**
 * @brief Multi-field first-order IMEX Euler stepper (SoA packs).
 *
 * Same attempt/commit isolation protocol as `ImexEulerStepper`, over
 * `std::array` scratch per field. For `N == 2`, `ExplicitRhs` should satisfy
 * `MultiStageFunction`. Solver bundles use
 * `std::make_tuple(std::ref(...), ...)`.
 *
 * @tparam ExplicitRhs Multi-field explicit RHS callable.
 * @tparam Solver      Injected `SolveFunction`-compatible solver.
 * @tparam N           Number of fields (`static_assert(N >= 1)`).
 */
template <class ExplicitRhs, class Solver, std::size_t N>
class MultiImexEulerStepper {
public:
  using ExplicitRhsType = ExplicitRhs;
  static constexpr std::size_t field_count = N;

  static_assert(N >= 1, "MultiImexEulerStepper requires N >= 1");

  MultiImexEulerStepper(double dt, std::array<std::size_t, N> local_sizes,
                        ExplicitRhs E, Solver solver,
                        pfc::sim::LinearOperatorDesc op_desc,
                        pfc::sim::SolveOptions opts = {})
      : m_dt(dt), m_E(std::move(E)), m_solver(std::move(solver)),
        m_op_desc(std::move(op_desc)), m_opts(std::move(opts)) {
    for (std::size_t i = 0; i < N; ++i) {
      m_u_work[i].assign(local_sizes[i], 0.0);
      m_e[i].assign(local_sizes[i], 0.0);
      m_rhs_vec[i].assign(local_sizes[i], 0.0);
      m_candidate[i].assign(local_sizes[i], 0.0);
      m_u_checkpoint[i].assign(local_sizes[i], 0.0);
    }
  }

  /**
   * @brief Attempt one multi-field IMEX Euler step without mutating accepted
   *        buffers.
   *
   * @note `ctx` precedes the field pack because a C++ parameter pack must be
   *       last; call as `attempt(t, ctx, u1, u2, ...)`.
   */
  template <class... U>
  [[nodiscard]] ImexStepAttempt attempt(double t, pfc::sim::StageContext &ctx,
                                        const std::vector<U> &...u_accepted) {
    static_assert(sizeof...(U) == N,
                  "MultiImexEulerStepper::attempt: buffer count must match N");
    static_assert((std::is_same_v<U, double> && ...),
                  "MultiImexEulerStepper requires std::vector<double>");
    m_last_success = false;
    copy_accepted_to_work(std::index_sequence_for<U...>{}, u_accepted...);
    auto u_pack = make_work_tuple(std::index_sequence_for<U...>{});
    auto e_pack = make_e_tuple(std::index_sequence_for<U...>{});
    m_E(t, u_pack, e_pack);
    form_rhs(std::index_sequence_for<U...>{}, u_accepted...);
    ctx.evaluation_time = t + m_dt;
    auto rhs_bundle = make_rhs_bundle(std::index_sequence_for<U...>{});
    auto candidate_bundle =
        make_candidate_bundle(std::index_sequence_for<U...>{});
    auto outcome =
        m_solver(m_op_desc, rhs_bundle, candidate_bundle, m_opts, ctx);
    if (outcome.status == pfc::sim::ConvergenceStatus::converged) {
      detail::ingest_multi_field_solution(m_candidate, outcome.solution);
      m_last_success = true;
      return detail::make_attempt_from_outcome(true, t + m_dt, outcome);
    }
    return detail::make_attempt_from_outcome(false, t + m_dt, outcome);
  }

  /**
   * @brief Commit candidates into accepted buffers only after a successful
   *        attempt.
   */
  template <class... U>
  [[nodiscard]] bool commit(std::vector<U> &...u_accepted) const {
    static_assert(sizeof...(U) == N,
                  "MultiImexEulerStepper::commit: buffer count must match N");
    static_assert((std::is_same_v<U, double> && ...),
                  "MultiImexEulerStepper requires std::vector<double>");
    if (!m_last_success) {
      return false;
    }
    copy_candidate_to_accepted(std::index_sequence_for<U...>{}, u_accepted...);
    return true;
  }

  [[nodiscard]] double dt() const noexcept { return m_dt; }

  template <class... U> void save_state(const std::vector<U> &...u_buffers) {
    static_assert(sizeof...(U) == N, "field count must match N");
    static_assert((std::is_same_v<U, double> && ...),
                  "checkpoint requires std::vector<double>");
    std::size_t i = 0;
    ((m_u_checkpoint[i++] = u_buffers), ...);
  }

  template <class... U> void restore_state(std::vector<U> &...u_buffers) {
    static_assert(sizeof...(U) == N, "field count must match N");
    static_assert((std::is_same_v<U, double> && ...),
                  "checkpoint requires std::vector<double>");
    std::size_t i = 0;
    ((u_buffers = m_u_checkpoint[i++]), ...);
  }

  [[nodiscard]] bool can_rollback() const noexcept { return true; }

private:
  template <std::size_t... I, class... U>
  void copy_accepted_to_work(std::index_sequence<I...>,
                             const std::vector<U> &...u_accepted) {
    ((m_u_work[I] = u_accepted), ...);
  }

  template <std::size_t... I> auto make_work_tuple(std::index_sequence<I...>) {
    return std::tie(m_u_work[I]...);
  }

  template <std::size_t... I> auto make_e_tuple(std::index_sequence<I...>) {
    return std::tie(m_e[I]...);
  }

  template <std::size_t... I, class... U>
  void form_rhs(std::index_sequence<I...>,
                const std::vector<U> &...u_accepted) {
    auto one = [this](std::vector<double> &rhs, const std::vector<double> &u,
                      const std::vector<double> &e) {
      for (std::size_t i = 0; i < u.size(); ++i) {
        rhs[i] = u[i] + m_dt * e[i];
      }
    };
    (one(m_rhs_vec[I], u_accepted, m_e[I]), ...);
  }

  template <std::size_t... I> auto make_rhs_bundle(std::index_sequence<I...>) {
    return std::make_tuple(std::ref(m_rhs_vec[I])...);
  }

  template <std::size_t... I>
  auto make_candidate_bundle(std::index_sequence<I...>) {
    return std::make_tuple(std::ref(m_candidate[I])...);
  }

  template <std::size_t... I, class... U>
  void copy_candidate_to_accepted(std::index_sequence<I...>,
                                  std::vector<U> &...u_accepted) const {
    ((u_accepted = m_candidate[I]), ...);
  }

  double m_dt{0.0};
  ExplicitRhs m_E;
  Solver m_solver;
  pfc::sim::LinearOperatorDesc m_op_desc;
  pfc::sim::SolveOptions m_opts;
  std::array<std::vector<double>, N> m_u_work;
  std::array<std::vector<double>, N> m_e;
  std::array<std::vector<double>, N> m_rhs_vec;
  std::array<std::vector<double>, N> m_candidate;
  std::array<std::vector<double>, N> m_u_checkpoint;
  bool m_last_success{false};
};

} // namespace pfc::sim::steppers
