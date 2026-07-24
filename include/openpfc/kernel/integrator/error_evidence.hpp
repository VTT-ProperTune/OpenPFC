// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file error_evidence.hpp
 * @brief Method-independent error evidence and controller normalization
 *
 * @details
 * Integrators produce method-family-agnostic `ErrorEvidence` on a step
 * attempt (embedded-pair norms, residual / a-posteriori norms, or a
 * documented method-specific extension hook). A thin
 * `reduce_error_evidence` helper may promote `AggregationScope::RankLocal`
 * evidence to `AlreadyReduced` (MPI_Allreduce MAX on norms; single-rank
 * identity). `normalize_error_evidence` consumes rank-consistent evidence
 * plus injected tolerances and returns a dimensionless metric with an
 * accept / reject / no-decision verdict — without computing a next `dt`.
 *
 * Transient per-step evidence is **not** checkpointed. Only future
 * controller history that affects subsequent decisions may persist
 * (restore is out of scope here).
 *
 * Non-scope for this seam: embedded RK stepper body (#162), tolerance /
 * step-bound JSON schema (#163), Simulator adaptive orchestration, and
 * on-hold `IntegratorResult` (#141).
 *
 * Producers collapse scalar or multi-field (N>=2) solution pairs /
 * residuals to per-field norms before calling the factories — this header
 * does not depend on `tuple_protocol`.
 *
 * @see docs/development/integrator_interface_contract.md §5
 * @see kernel/integrator/stage_context.hpp
 * @see kernel/integrator/workspace.hpp
 */

#include <mpi.h>

#include <cmath>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>

namespace pfc::integrator {

/**
 * @brief Kind of adaptive-step error evidence produced by an integrator
 *
 * Accept/reject math in `normalize_error_evidence` does **not** branch on
 * this enum — kinds identify how the producer obtained the norms.
 */
enum class EvidenceKind {
  EmbeddedPair,        ///< |y_high - y_low| (or equivalent) norms
  ResidualAPosteriori, ///< Residual / a-posteriori norms
  MethodSpecific       ///< Documented extension hook (e.g. step-doubling)
};

/**
 * @brief Whether field norms are local to this rank or already reduced
 */
enum class AggregationScope {
  RankLocal,      ///< Norms may differ across ranks; need reduce
  AlreadyReduced  ///< Rank-consistent; must not be reduced again
};

/**
 * @brief Method-independent error evidence for one step attempt
 *
 * @note Transient: not part of checkpoint state.
 */
struct ErrorEvidence {
  EvidenceKind kind{};
  AggregationScope scope{AggregationScope::RankLocal};
  bool valid{false}; ///< Global validity; false ⇒ NoDecision on normalize
  std::vector<bool> field_valid;   ///< Per-field; size == field_norms.size()
  std::vector<double> field_norms; ///< Non-negative per-field error/residual norms
  std::optional<double> combined_metric; ///< Optional pre-combined scalar
  std::optional<int> order_tag;          ///< Estimated method order (e.g. 3)
  std::optional<std::vector<double>> weights; ///< Optional per-field scales
};

/**
 * @brief Absolute / relative tolerances injected by the controller (#163)
 *
 * No JSON schema here — callers (tests or future config consumers) supply
 * values.
 */
struct ErrorTolerances {
  double absolute{0.0}; ///< atol
  double relative{0.0}; ///< rtol
};

/**
 * @brief Accept / reject / unavailable-estimator outcome for a step attempt
 *
 * Prefer this enum over a `bool accepted` member (avoids naming collisions
 * with static factories).
 */
enum class StepAttemptVerdict { Accept, Reject, NoDecision };

/**
 * @brief Normalized error metric and verdict (no next-dt recommendation)
 *
 * On a valid path, `metric <= 1.0` means Accept. Controllers must not
 * advance time on `Reject` or `NoDecision`.
 */
struct NormalizedError {
  double metric{0.0}; ///< Dimensionless; NaN when decision unavailable
  StepAttemptVerdict verdict{StepAttemptVerdict::NoDecision};
  bool decision_available{false}; ///< false iff invalid / unavailable estimator
};

namespace detail {

[[nodiscard]] inline bool norms_are_finite_nonnegative(std::span<const double> norms) {
  for (double v : norms) {
    if (!std::isfinite(v) || v < 0.0) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool weights_match(std::span<const double> norms,
                                        std::optional<std::span<const double>> weights) {
  if (!weights.has_value()) {
    return true;
  }
  if (weights->size() != norms.size()) {
    return false;
  }
  return norms_are_finite_nonnegative(*weights);
}

[[nodiscard]] inline ErrorEvidence
make_evidence(EvidenceKind kind, std::span<const double> field_error_norms,
              AggregationScope scope, std::optional<int> order_tag,
              std::optional<std::span<const double>> weights) {
  if (field_error_norms.empty() || !norms_are_finite_nonnegative(field_error_norms) ||
      !weights_match(field_error_norms, weights)) {
    ErrorEvidence invalid;
    invalid.kind = kind;
    invalid.scope = scope;
    invalid.valid = false;
    invalid.order_tag = order_tag;
    return invalid;
  }

  ErrorEvidence ev;
  ev.kind = kind;
  ev.scope = scope;
  ev.valid = true;
  ev.field_norms.assign(field_error_norms.begin(), field_error_norms.end());
  ev.field_valid.assign(field_error_norms.size(), true);
  ev.order_tag = order_tag;
  if (weights.has_value()) {
    ev.weights = std::vector<double>(weights->begin(), weights->end());
  }
  return ev;
}

[[nodiscard]] inline NormalizedError make_no_decision() {
  return NormalizedError{.metric = std::numeric_limits<double>::quiet_NaN(),
                         .verdict = StepAttemptVerdict::NoDecision,
                         .decision_available = false};
}

} // namespace detail

/**
 * @brief Build embedded-pair evidence from per-field error norms
 *
 * @param field_error_norms Non-empty span of non-negative finite norms
 *        (length 1 = scalar; N>=2 = multi-field)
 * @param scope Aggregation scope declared by the producer
 * @param order_tag Optional estimated method order
 * @param weights Optional per-field scales (must match norms size)
 * @return Valid evidence, or `valid=false` if inputs are empty / invalid
 */
[[nodiscard]] inline ErrorEvidence make_embedded_pair_evidence(
    std::span<const double> field_error_norms, AggregationScope scope,
    std::optional<int> order_tag = {},
    std::optional<std::span<const double>> weights = {}) {
  return detail::make_evidence(EvidenceKind::EmbeddedPair, field_error_norms, scope,
                               order_tag, weights);
}

/**
 * @brief Build residual / a-posteriori evidence from per-field norms
 *
 * Same shape as `make_embedded_pair_evidence` with
 * `EvidenceKind::ResidualAPosteriori`.
 */
[[nodiscard]] inline ErrorEvidence make_residual_evidence(
    std::span<const double> field_error_norms, AggregationScope scope,
    std::optional<int> order_tag = {},
    std::optional<std::span<const double>> weights = {}) {
  return detail::make_evidence(EvidenceKind::ResidualAPosteriori, field_error_norms,
                               scope, order_tag, weights);
}

/**
 * @brief Documented extension hook for method-specific estimators
 *
 * Controllers must not branch on `EvidenceKind` when normalizing — this
 * factory only labels the producer path (e.g. synthetic step-doubling
 * fixtures in tests).
 */
[[nodiscard]] inline ErrorEvidence make_method_specific_evidence(
    std::span<const double> field_error_norms, AggregationScope scope,
    std::optional<int> order_tag = {},
    std::optional<std::span<const double>> weights = {}) {
  return detail::make_evidence(EvidenceKind::MethodSpecific, field_error_norms, scope,
                               order_tag, weights);
}

/**
 * @brief Invalid / unavailable estimator evidence for a given kind
 *
 * `normalize_error_evidence` returns `NoDecision` with
 * `decision_available=false`. Controllers must not advance time.
 */
[[nodiscard]] inline ErrorEvidence make_invalid_evidence(EvidenceKind kind) {
  ErrorEvidence ev;
  ev.kind = kind;
  ev.scope = AggregationScope::RankLocal;
  ev.valid = false;
  return ev;
}

/**
 * @brief Promote rank-local evidence to already-reduced (or leave unchanged)
 *
 * - Invalid evidence is returned unchanged.
 * - `AlreadyReduced` is returned unchanged (no double-reduce).
 * - `RankLocal` with communicator size 1 sets `AlreadyReduced` and leaves
 *   norms unchanged (identity).
 * - `RankLocal` with size > 1 performs `MPI_Allreduce` MAX on each
 *   `field_norms[i]` and on `combined_metric` if present, AND-reduces
 *   validity flags, then sets `AlreadyReduced`.
 *
 * @param ev Evidence to reduce (taken by value)
 * @param comm MPI communicator (default world)
 */
[[nodiscard]] inline ErrorEvidence
reduce_error_evidence(ErrorEvidence ev, MPI_Comm comm = MPI_COMM_WORLD) {
  if (!ev.valid) {
    return ev;
  }
  if (ev.scope == AggregationScope::AlreadyReduced) {
    return ev;
  }

  int size = 1;
  int err = MPI_Comm_size(comm, &size);
  pfc::mpi::throw_on_mpi_error(err, "MPI_Comm_size in reduce_error_evidence");
  if (size <= 1) {
    ev.scope = AggregationScope::AlreadyReduced;
    return ev;
  }

  // Validity: AND across ranks (pack bools as ints).
  int global_valid = ev.valid ? 1 : 0;
  err = MPI_Allreduce(MPI_IN_PLACE, &global_valid, 1, MPI_INT, MPI_LAND, comm);
  pfc::mpi::throw_on_mpi_error(err, "MPI_Allreduce for global_valid in reduce_error_evidence");
  ev.valid = (global_valid != 0);

  if (!ev.field_norms.empty()) {
    err = MPI_Allreduce(MPI_IN_PLACE, ev.field_norms.data(),
                  static_cast<int>(ev.field_norms.size()), MPI_DOUBLE, MPI_MAX, comm);
    pfc::mpi::throw_on_mpi_error(err, "MPI_Allreduce for field_norms in reduce_error_evidence");
  }

  if (!ev.field_valid.empty()) {
    std::vector<int> packed(ev.field_valid.size());
    for (std::size_t i = 0; i < ev.field_valid.size(); ++i) {
      packed[i] = ev.field_valid[i] ? 1 : 0;
    }
    err = MPI_Allreduce(MPI_IN_PLACE, packed.data(), static_cast<int>(packed.size()), MPI_INT,
                  MPI_LAND, comm);
    pfc::mpi::throw_on_mpi_error(err, "MPI_Allreduce for field_valid in reduce_error_evidence");
    for (std::size_t i = 0; i < packed.size(); ++i) {
      ev.field_valid[i] = (packed[i] != 0);
    }
  }

  if (ev.combined_metric.has_value()) {
    double combined = *ev.combined_metric;
    err = MPI_Allreduce(MPI_IN_PLACE, &combined, 1, MPI_DOUBLE, MPI_MAX, comm);
    pfc::mpi::throw_on_mpi_error(err, "MPI_Allreduce for combined_metric in reduce_error_evidence");
    ev.combined_metric = combined;
  }

  ev.scope = AggregationScope::AlreadyReduced;
  return ev;
}

/**
 * @brief Normalize rank-consistent evidence into a metric and verdict
 *
 * Prefers `scope == AlreadyReduced` (or RankLocal when the caller
 * guarantees single-rank / already-consistent data). Does **not** switch
 * on `EvidenceKind`. Does **not** compute or return a next `dt`.
 *
 * Valid path: for each field i,
 * `e_i = field_norms[i] / (atol + rtol * scale_i)` with
 * `scale_i = weights[i]` if present else `1.0`. Metric is the max-norm of
 * `e_i`. Accept iff `metric <= 1.0`, else Reject. If `den == 0` for any
 * field, treat as Reject (infinite error).
 *
 * If `field_norms` is empty but `combined_metric` is set, fall back to
 * `combined_metric / (atol + rtol)`.
 *
 * Invalid / unavailable path: `verdict = NoDecision`,
 * `decision_available = false`, `metric = NaN`.
 *
 * @param ev Evidence (prefer AlreadyReduced)
 * @param tol Injected absolute and relative tolerances
 */
[[nodiscard]] inline NormalizedError
normalize_error_evidence(const ErrorEvidence &ev, const ErrorTolerances &tol) {
  if (!ev.valid) {
    return detail::make_no_decision();
  }
  for (bool fv : ev.field_valid) {
    if (!fv) {
      return detail::make_no_decision();
    }
  }
  if (ev.field_norms.empty() && !ev.combined_metric.has_value()) {
    return detail::make_no_decision();
  }

  double metric = 0.0;

  if (!ev.field_norms.empty()) {
    if (ev.weights.has_value() && ev.weights->size() != ev.field_norms.size()) {
      return detail::make_no_decision();
    }
    for (std::size_t i = 0; i < ev.field_norms.size(); ++i) {
      const double scale =
          (ev.weights.has_value()) ? (*ev.weights)[i] : 1.0;
      const double den = tol.absolute + tol.relative * scale;
      if (den == 0.0) {
        return NormalizedError{.metric = std::numeric_limits<double>::infinity(),
                               .verdict = StepAttemptVerdict::Reject,
                               .decision_available = true};
      }
      metric = std::max(metric, ev.field_norms[i] / den);
    }
  } else {
    const double den = tol.absolute + tol.relative;
    if (den == 0.0) {
      return NormalizedError{.metric = std::numeric_limits<double>::infinity(),
                             .verdict = StepAttemptVerdict::Reject,
                             .decision_available = true};
    }
    metric = *ev.combined_metric / den;
  }

  const auto verdict =
      (metric <= 1.0) ? StepAttemptVerdict::Accept : StepAttemptVerdict::Reject;
  return NormalizedError{.metric = metric, .verdict = verdict, .decision_available = true};
}

} // namespace pfc::integrator
