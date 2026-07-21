// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file euler_attempt.hpp
 * @brief Explicit-Euler proof path through the shared step-attempt seam.
 *
 * @details
 * `EulerAttemptStepper` writes the candidate `u + dt * du` into method-owned
 * storage. The accepted input buffer is never mutated; the driver must call
 * `commit_step_attempt` to apply a successful candidate.
 *
 * Soft failure (`prep` or `eval` returns `false`) reports `success == false`,
 * leaves accepted state unchanged, and leaves workspace **reusable**
 * (`workspace_reusable() == true`; buffers remain allocated/sized, contents
 * unspecified). Hard errors (size mismatch) throw before any candidate write.
 *
 * Legacy in-place `EulerStepper::step` in `euler.hpp` is untouched.
 *
 * Adapter note: existing `StageFunction` / factory lambdas take non-const `u`
 * and return void. Wrap them for this path, e.g.
 * `[&](const StageContext &ctx, const auto &u, auto &du) -> bool {
 *    rhs(ctx.time, const_cast<std::vector<double>&>(u), du); return true; }`
 * — prefer a dedicated bool-returning evaluator when practical. Do not change
 * the `create()` factories in `euler.hpp`.
 *
 * @see step_attempt.hpp for `StepAttemptResult`, concepts, and commit helpers
 */

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/kernel/integrator/stage_context.hpp>
#include <openpfc/kernel/simulation/steppers/step_attempt.hpp>

namespace pfc::sim::steppers {

/**
 * @brief Explicit forward-Euler step-attempt stepper (isolated candidate).
 *
 * @tparam OpEval Operator evaluator (`OperatorEvaluator`).
 * @tparam Prep   Preparation service (`PreparationService`).
 */
template <class OpEval, class Prep>
  requires OperatorEvaluator<OpEval> && PreparationService<Prep>
class EulerAttemptStepper {
public:
  /**
   * @brief Allocate method-owned `du` and candidate buffers of `local_size`.
   */
  explicit EulerAttemptStepper(std::size_t local_size)
      : m_local_size(local_size), m_du(local_size, 0.0),
        m_candidate(local_size, 0.0), m_workspace_reusable(true) {}

  /**
   * @brief Attempt one Euler step without mutating accepted state.
   *
   * @param t    Current accepted time.
   * @param dt   Proposed step size.
   * @param u    Accepted state (read-only; never written).
   * @param eval Operator evaluator; fills `du`, returns false on soft fail.
   * @param prep Preparation hook; returns false on soft fail.
   * @return Step-attempt result with candidate view into method-owned storage.
   *
   * @throws std::invalid_argument if `u.size() != local_size`.
   */
  [[nodiscard]] StepAttemptResult attempt(double t, double dt,
                                          const std::vector<double> &u,
                                          OpEval &eval, Prep &prep) {
    if (u.size() != m_local_size) {
      throw std::invalid_argument(
          "EulerAttemptStepper::attempt: u.size() (" +
          std::to_string(u.size()) + ") != local_size (" +
          std::to_string(m_local_size) + ")");
    }

    const pfc::integrator::StageContext ctx{
        .time = t,
        .dt = dt,
        .stage_index = 0,
        .region_kind = pfc::integrator::StageContext::RegionKind::All,
        .needs_boundary_update = false,
        .needs_halo_exchange = false,
    };

    if (!prep(ctx)) {
      m_workspace_reusable = true;
      return StepAttemptResult(t, dt, t, /*success=*/false, m_candidate);
    }

    if (!eval(ctx, u, m_du)) {
      m_workspace_reusable = true;
      return StepAttemptResult(t, dt, t, /*success=*/false, m_candidate);
    }

    for (std::size_t i = 0; i < m_local_size; ++i) {
      m_candidate[i] = u[i] + dt * m_du[i];
    }
    m_workspace_reusable = true;
    return StepAttemptResult(t, dt, t + dt, /*success=*/true, m_candidate);
  }

  /**
   * @brief Whether method-owned workspace is reusable after the last attempt.
   *
   * Soft failure leaves workspace reusable (allocated/sized; contents
   * unspecified). Success also leaves it reusable for the next attempt.
   */
  [[nodiscard]] bool workspace_reusable() const noexcept {
    return m_workspace_reusable;
  }

  [[nodiscard]] std::size_t local_size() const noexcept { return m_local_size; }

private:
  std::size_t m_local_size{0};
  std::vector<double> m_du;
  std::vector<double> m_candidate;
  bool m_workspace_reusable{true};
};

/**
 * @brief Two-field explicit-Euler step-attempt stepper (isolated candidates).
 *
 * Same isolation / soft-failure / commit rules as `EulerAttemptStepper`, with
 * one candidate buffer per field.
 *
 * @tparam OpEval `MultiOperatorEvaluator2`
 * @tparam Prep   `PreparationService`
 */
template <class OpEval, class Prep>
  requires MultiOperatorEvaluator2<OpEval> && PreparationService<Prep>
class MultiEulerAttemptStepper {
public:
  static constexpr std::size_t field_count = 2;

  explicit MultiEulerAttemptStepper(std::array<std::size_t, 2> local_sizes)
      : m_local_sizes(local_sizes), m_workspace_reusable(true) {
    for (std::size_t f = 0; f < 2; ++f) {
      m_du[f].assign(local_sizes[f], 0.0);
      m_candidate[f].assign(local_sizes[f], 0.0);
    }
  }

  /**
   * @brief Attempt one multi-field Euler step without mutating accepted state.
   *
   * @throws std::invalid_argument on per-field size mismatch.
   */
  [[nodiscard]] MultiStepAttemptResult<2>
  attempt(double t, double dt, const std::vector<double> &u0,
          const std::vector<double> &u1, OpEval &eval, Prep &prep) {
    if (u0.size() != m_local_sizes[0] || u1.size() != m_local_sizes[1]) {
      throw std::invalid_argument(
          "MultiEulerAttemptStepper::attempt: accepted buffer size mismatch "
          "(u0.size=" +
          std::to_string(u0.size()) + " expected " +
          std::to_string(m_local_sizes[0]) + "; u1.size=" +
          std::to_string(u1.size()) + " expected " +
          std::to_string(m_local_sizes[1]) + ")");
    }

    const pfc::integrator::StageContext ctx{
        .time = t,
        .dt = dt,
        .stage_index = 0,
        .region_kind = pfc::integrator::StageContext::RegionKind::All,
        .needs_boundary_update = false,
        .needs_halo_exchange = false,
    };

    const std::array<const std::vector<double> *, 2> cand_ptrs{&m_candidate[0],
                                                               &m_candidate[1]};

    if (!prep(ctx)) {
      m_workspace_reusable = true;
      return MultiStepAttemptResult<2>(t, dt, t, /*success=*/false, cand_ptrs);
    }

    if (!eval(ctx, u0, u1, m_du[0], m_du[1])) {
      m_workspace_reusable = true;
      return MultiStepAttemptResult<2>(t, dt, t, /*success=*/false, cand_ptrs);
    }

    for (std::size_t i = 0; i < m_local_sizes[0]; ++i) {
      m_candidate[0][i] = u0[i] + dt * m_du[0][i];
    }
    for (std::size_t i = 0; i < m_local_sizes[1]; ++i) {
      m_candidate[1][i] = u1[i] + dt * m_du[1][i];
    }
    m_workspace_reusable = true;
    return MultiStepAttemptResult<2>(t, dt, t + dt, /*success=*/true,
                                     cand_ptrs);
  }

  [[nodiscard]] bool workspace_reusable() const noexcept {
    return m_workspace_reusable;
  }

private:
  std::array<std::size_t, 2> m_local_sizes{};
  std::array<std::vector<double>, 2> m_du;
  std::array<std::vector<double>, 2> m_candidate;
  bool m_workspace_reusable{true};
};

} // namespace pfc::sim::steppers
