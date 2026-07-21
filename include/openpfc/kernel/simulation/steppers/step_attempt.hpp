// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file step_attempt.hpp
 * @brief Shared step-attempt result type, invocation concepts, and commit helper.
 *
 * @details
 * The step-attempt seam is the driver-facing contract for methods that must
 * isolate a candidate state from the accepted input buffer until an explicit
 * commit. Soft failure (`success == false`) leaves the accepted buffer
 * bitwise unchanged and leaves method-owned workspace **reusable** (buffers
 * remain allocated and sized; contents are unspecified).
 *
 * Candidate views in `StepAttemptResult` / `MultiStepAttemptResult` bind to
 * method-owned storage and remain valid until the next `attempt()` on the
 * owning stepper or until that stepper is destroyed (same lifetime rule as
 * `EmbeddedStepAttemptResult`).
 *
 * Distinct from `EmbeddedStepAttemptResult` (embedded high/low/error pair
 * evidence). This type is the shared single-candidate (or N-field) shape for
 * Euler, and later IMEX/ETD leaves.
 *
 * @note Uses `pfc::integrator::StageContext` from
 *       `openpfc/kernel/integrator/stage_context.hpp` — not
 *       `pfc::sim::StageContext` in `solver_contract.hpp`.
 *
 * @see euler_attempt.hpp for the explicit-Euler proof path
 * @see embedded_rk.hpp for the orthogonal embedded-pair attempt API
 */

#include <array>
#include <concepts>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <openpfc/kernel/integrator/stage_context.hpp>

namespace pfc::sim::steppers {

/**
 * @brief Outcome of one method step attempt with an isolated candidate.
 *
 * `candidate` is a view into method-owned storage. It is valid until the next
 * `attempt()` on the owning stepper or stepper destruction. On soft failure,
 * `t1` equals `t0`; on success, `t1 == t0 + dt`.
 */
struct StepAttemptResult {
  double t0{};
  double dt{};
  double t1{}; ///< `t0 + dt` on success; `t0` on soft failure
  bool success{false};
  const std::vector<double> &candidate;
  std::optional<double> error_norm{};  ///< stub for future error-evidence taxonomy
  std::optional<double> min_next_dt{}; ///< stub for future next-attempt constraints

  StepAttemptResult(double t0_in, double dt_in, double t1_in, bool success_in,
                    const std::vector<double> &candidate_in,
                    std::optional<double> error_norm_in = std::nullopt,
                    std::optional<double> min_next_dt_in = std::nullopt)
      : t0(t0_in), dt(dt_in), t1(t1_in), success(success_in),
        candidate(candidate_in), error_norm(std::move(error_norm_in)),
        min_next_dt(std::move(min_next_dt_in)) {}
};

/**
 * @brief N-field step-attempt outcome with one isolated candidate per field.
 *
 * Candidate references bind to method-owned buffers with the same lifetime
 * rule as `StepAttemptResult::candidate`.
 */
template <std::size_t N> struct MultiStepAttemptResult {
  static_assert(N >= 1, "MultiStepAttemptResult requires N >= 1");

  double t0{};
  double dt{};
  double t1{};
  bool success{false};
  std::array<const std::vector<double> *, N> candidates{};
  std::optional<double> error_norm{};
  std::optional<double> min_next_dt{};

  MultiStepAttemptResult(
      double t0_in, double dt_in, double t1_in, bool success_in,
      std::array<const std::vector<double> *, N> candidates_in,
      std::optional<double> error_norm_in = std::nullopt,
      std::optional<double> min_next_dt_in = std::nullopt)
      : t0(t0_in), dt(dt_in), t1(t1_in), success(success_in),
        candidates(candidates_in), error_norm(std::move(error_norm_in)),
        min_next_dt(std::move(min_next_dt_in)) {}

  [[nodiscard]] const std::vector<double> &candidate(std::size_t i) const {
    return *candidates[i];
  }
};

/**
 * @brief Operator evaluator for the step-attempt path (read-only accepted state).
 *
 * Invocable as `eval(ctx, u, du) -> bool`. Must fill `du` and must not write
 * `u`. Returns `true` on success, `false` for a soft (recoverable) failure.
 *
 * Distinct from `StageFunction` (`rhs(t, u, du)` with non-const `u` and void
 * return) so the attempt path can enforce read-only accepted state. Thin
 * adapters that wrap a void RHS and always return `true` are fine at call
 * sites; do not change legacy `EulerStepper` / `create` factories.
 */
template <class E>
concept OperatorEvaluator = requires(
    E eval, const pfc::integrator::StageContext &ctx,
    const std::vector<double> &u, std::vector<double> &du) {
  { eval(ctx, u, du) } -> std::convertible_to<bool>;
};

/**
 * @brief Preparation / stage-context service hook for the step-attempt path.
 *
 * Invocable as `prep(ctx) -> bool`. Returns `true` on success, `false` for a
 * soft failure (e.g. forced prep failure in tests). May be a no-op that
 * always returns `true`.
 */
template <class P>
concept PreparationService =
    requires(P prep, const pfc::integrator::StageContext &ctx) {
      { prep(ctx) } -> std::convertible_to<bool>;
    };

/**
 * @brief Two-field operator evaluator (const accepted buffers).
 *
 * Invocable as `eval(ctx, u0, u1, du0, du1) -> bool`.
 */
template <class E>
concept MultiOperatorEvaluator2 = requires(
    E eval, const pfc::integrator::StageContext &ctx,
    const std::vector<double> &u0, const std::vector<double> &u1,
    std::vector<double> &du0, std::vector<double> &du1) {
  { eval(ctx, u0, u1, du0, du1) } -> std::convertible_to<bool>;
};

/**
 * @brief Copy a successful candidate into the accepted buffer.
 *
 * @throws std::invalid_argument if `!result.success` (misuse is loud).
 */
[[nodiscard]] inline void
commit_step_attempt(std::vector<double> &accepted,
                    const StepAttemptResult &result) {
  if (!result.success) {
    throw std::invalid_argument(
        "commit_step_attempt: cannot commit a failed StepAttemptResult "
        "(success == false)");
  }
  accepted = result.candidate;
}

/**
 * @brief Copy two successful candidates into accepted field buffers (N=2).
 *
 * @throws std::invalid_argument if `!result.success`.
 */
[[nodiscard]] inline void
commit_step_attempt(std::vector<double> &accepted0,
                    std::vector<double> &accepted1,
                    const MultiStepAttemptResult<2> &result) {
  if (!result.success) {
    throw std::invalid_argument(
        "commit_step_attempt: cannot commit a failed MultiStepAttemptResult "
        "(success == false)");
  }
  accepted0 = result.candidate(0);
  accepted1 = result.candidate(1);
}

} // namespace pfc::sim::steppers
