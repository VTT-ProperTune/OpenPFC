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
 * retired accept/reject DTO (#141); the live type is `StepAttemptResult`.
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
  RankLocal,     ///< Norms may differ across ranks; need reduce
  AlreadyReduced ///< Rank-consistent; must not be reduced again
};

/**
 * @brief Where the solution scale in `atol + rtol * scale` comes from
 *
 * `Unspecified` does not mean 1. Relative tolerances then have no scale
 * and normalization fails closed. `Unit` is the explicit choice that
 * `rtol` is relative to 1. `Supplied` reads `weights`.
 */
enum class SolutionScale { Unspecified, Unit, Supplied };

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
  /// Per-field solution scales. Required when `solution_scale` is `Supplied`.
  /// For a distributed field this is the same norm as `field_norms` (max-abs).
  std::optional<std::vector<double>> weights;
  SolutionScale solution_scale{SolutionScale::Unspecified};
};

/**
 * @brief Absolute / relative tolerances injected by the controller (#163)
 *
 * No JSON schema here — callers (tests or future config consumers) supply
 * values.
 */
struct ErrorTolerances {
  double absolute{0.0}; ///< atol, used when `absolute_per_field` is empty
  double relative{0.0}; ///< rtol, used when `relative_per_field` is empty
  /// When set, length must equal the evidence field count.
  std::optional<std::vector<double>> absolute_per_field;
  std::optional<std::vector<double>> relative_per_field;
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

[[nodiscard]] inline bool
norms_are_finite_nonnegative(std::span<const double> norms) {
  for (double v : norms) {
    if (!std::isfinite(v) || v < 0.0) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool
weights_match(std::span<const double> norms,
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
              std::optional<std::span<const double>> weights, SolutionScale scale) {
  // Weights are the solution scale. They override a Unit/Unspecified request.
  if (weights.has_value()) {
    scale = SolutionScale::Supplied;
  }
  if (scale == SolutionScale::Supplied && !weights.has_value()) {
    ErrorEvidence invalid;
    invalid.kind = kind;
    invalid.scope = scope;
    invalid.valid = false;
    invalid.order_tag = order_tag;
    invalid.solution_scale = SolutionScale::Supplied;
    return invalid;
  }
  if (field_error_norms.empty() ||
      !norms_are_finite_nonnegative(field_error_norms) ||
      !weights_match(field_error_norms, weights)) {
    ErrorEvidence invalid;
    invalid.kind = kind;
    invalid.scope = scope;
    invalid.valid = false;
    invalid.order_tag = order_tag;
    invalid.solution_scale = scale;
    return invalid;
  }

  ErrorEvidence ev;
  ev.kind = kind;
  ev.scope = scope;
  ev.valid = true;
  ev.field_norms.assign(field_error_norms.begin(), field_error_norms.end());
  ev.field_valid.assign(field_error_norms.size(), true);
  ev.order_tag = order_tag;
  ev.solution_scale = scale;
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
[[nodiscard]] inline ErrorEvidence
make_embedded_pair_evidence(std::span<const double> field_error_norms,
                            AggregationScope scope,
                            std::optional<int> order_tag = {},
                            std::optional<std::span<const double>> weights = {},
                            SolutionScale scale = SolutionScale::Unspecified) {
  return detail::make_evidence(EvidenceKind::EmbeddedPair, field_error_norms, scope,
                               order_tag, weights, scale);
}

/**
 * @brief Build residual / a-posteriori evidence from per-field norms
 *
 * Same shape as `make_embedded_pair_evidence` with
 * `EvidenceKind::ResidualAPosteriori`.
 */
[[nodiscard]] inline ErrorEvidence
make_residual_evidence(std::span<const double> field_error_norms,
                       AggregationScope scope, std::optional<int> order_tag = {},
                       std::optional<std::span<const double>> weights = {},
                       SolutionScale scale = SolutionScale::Unspecified) {
  return detail::make_evidence(EvidenceKind::ResidualAPosteriori, field_error_norms,
                               scope, order_tag, weights, scale);
}

/**
 * @brief Documented extension hook for method-specific estimators
 *
 * Controllers must not branch on `EvidenceKind` when normalizing — this
 * factory only labels the producer path (e.g. synthetic step-doubling
 * fixtures in tests).
 */
[[nodiscard]] inline ErrorEvidence
make_method_specific_evidence(std::span<const double> field_error_norms,
                              AggregationScope scope,
                              std::optional<int> order_tag = {},
                              std::optional<std::span<const double>> weights = {},
                              SolutionScale scale = SolutionScale::Unspecified) {
  return detail::make_evidence(EvidenceKind::MethodSpecific, field_error_norms,
                               scope, order_tag, weights, scale);
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
 * - `AlreadyReduced` is returned unchanged (no double-reduce). Callers on
 *   one communicator pass the same scope.
 * - Invalid evidence on one rank is returned unchanged.
 * - `RankLocal` with communicator size 1 sets `AlreadyReduced` and leaves
 *   norms unchanged (identity) when the evidence is valid.
 * - `RankLocal` with size > 1 first reduces a fixed metadata record: field
 *   counts, weight presence, scale kind, and the validity bit. Every rank
 *   enters that reduction, including a rank whose evidence is already
 *   invalid. If any rank is invalid, or the records disagree, every rank
 *   receives `valid=false` and `AlreadyReduced` and no field buffer is
 *   reduced. Otherwise `MPI_Allreduce` MAX combines each `field_norms[i]`,
 *   each supplied scale, and `combined_metric` if present, AND-reduces
 *   per-field validity, then sets `AlreadyReduced`.
 *
 * @param ev Evidence to reduce (taken by value)
 * @param comm MPI communicator (default world)
 */
[[nodiscard]] inline ErrorEvidence
reduce_error_evidence(ErrorEvidence ev, MPI_Comm comm = MPI_COMM_WORLD) {
  if (ev.scope == AggregationScope::AlreadyReduced) {
    return ev;
  }

  int size = 1;
  int err = MPI_Comm_size(comm, &size);
  pfc::mpi::throw_on_mpi_error(err, "MPI_Comm_size in reduce_error_evidence");
  if (size <= 1) {
    if (ev.valid) {
      ev.scope = AggregationScope::AlreadyReduced;
    }
    return ev;
  }

  // Counts, scale kind, and validity travel in one fixed record, before any
  // variable-length buffer. A mismatch or an invalid rank fails closed on
  // every rank. Reducing the buffers first would be undefined when the
  // lengths differ, and an invalid rank must not skip the collective.
  const int meta_local[7] = {
      static_cast<int>(ev.field_norms.size()),
      static_cast<int>(ev.field_valid.size()),
      ev.combined_metric.has_value() ? 1 : 0,
      ev.weights.has_value() ? 1 : 0,
      ev.weights.has_value() ? static_cast<int>(ev.weights->size()) : 0,
      static_cast<int>(ev.solution_scale),
      ev.valid ? 1 : 0,
  };
  int meta_min[7] = {};
  int meta_max[7] = {};
  err = MPI_Allreduce(meta_local, meta_min, 7, MPI_INT, MPI_MIN, comm);
  pfc::mpi::throw_on_mpi_error(err, "MPI_Allreduce for error-evidence shape min");
  err = MPI_Allreduce(meta_local, meta_max, 7, MPI_INT, MPI_MAX, comm);
  pfc::mpi::throw_on_mpi_error(err, "MPI_Allreduce for error-evidence shape max");
  bool metadata_agrees = true;
  for (int i = 0; i < 6; ++i) {
    if (meta_min[i] != meta_max[i]) {
      metadata_agrees = false;
      break;
    }
  }
  if (!metadata_agrees || meta_min[6] == 0) {
    ev.valid = false;
    ev.scope = AggregationScope::AlreadyReduced;
    return ev;
  }

  if (!ev.field_norms.empty()) {
    err = MPI_Allreduce(MPI_IN_PLACE, ev.field_norms.data(),
                        static_cast<int>(ev.field_norms.size()), MPI_DOUBLE, MPI_MAX,
                        comm);
    pfc::mpi::throw_on_mpi_error(
        err, "MPI_Allreduce for field_norms in reduce_error_evidence");
  }

  if (!ev.field_valid.empty()) {
    std::vector<int> packed(ev.field_valid.size());
    for (std::size_t i = 0; i < ev.field_valid.size(); ++i) {
      packed[i] = ev.field_valid[i] ? 1 : 0;
    }
    err = MPI_Allreduce(MPI_IN_PLACE, packed.data(), static_cast<int>(packed.size()),
                        MPI_INT, MPI_LAND, comm);
    pfc::mpi::throw_on_mpi_error(
        err, "MPI_Allreduce for field_valid in reduce_error_evidence");
    for (std::size_t i = 0; i < packed.size(); ++i) {
      ev.field_valid[i] = (packed[i] != 0);
    }
  }

  if (ev.combined_metric.has_value()) {
    double combined = *ev.combined_metric;
    err = MPI_Allreduce(MPI_IN_PLACE, &combined, 1, MPI_DOUBLE, MPI_MAX, comm);
    pfc::mpi::throw_on_mpi_error(
        err, "MPI_Allreduce for combined_metric in reduce_error_evidence");
    ev.combined_metric = combined;
  }

  // Supplied scales use the same aggregate as the field norms (local
  // max-abs). The global scale is the max across ranks, then the metric is
  // ||e||_inf / (atol + rtol * ||y||_inf).
  if (ev.weights.has_value()) {
    err = MPI_Allreduce(MPI_IN_PLACE, ev.weights->data(),
                        static_cast<int>(ev.weights->size()), MPI_DOUBLE, MPI_MAX,
                        comm);
    pfc::mpi::throw_on_mpi_error(err, "MPI_Allreduce for solution scales");
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
 * `e_i = field_norms[i] / (atol_i + rtol_i * scale_i)`.
 * Metric is the max of `e_i`. Accept iff `metric <= 1.0`, else Reject.
 * If `den == 0` for any field, treat as Reject (infinite error).
 *
 * `scale_i` is never the constant 1 unless `solution_scale` is `Unit`.
 * `Unspecified` is legal only for pure absolute tolerances (`rtol_i == 0`).
 * `Supplied` uses `weights[i]`, the same aggregate the producer used for
 * `field_norms` (for a distributed field, both are local max-abs values;
 * `reduce_error_evidence` takes the global max of each). A relative
 * tolerance with no explicit scale yields `NoDecision`.
 *
 * If `field_norms` is empty but `combined_metric` is set, that scalar is
 * one field and uses the same scale rule.
 *
 * Invalid / unavailable path: `verdict = NoDecision`,
 * `decision_available = false`, `metric = NaN`.
 *
 * @param ev Evidence (prefer AlreadyReduced)
 * @param tol Injected absolute and relative tolerances
 */
namespace detail {

enum class DenomKind { Ok, NoDecision, ZeroDenominator };

struct Denominator {
  DenomKind kind{DenomKind::NoDecision};
  double value{0.0};
};

[[nodiscard]] inline Denominator field_denominator(const ErrorEvidence &ev,
                                                   const ErrorTolerances &tol,
                                                   std::size_t index,
                                                   std::size_t count) {
  auto pick = [&](double scalar, const std::optional<std::vector<double>> &per_field,
                  double &out) -> bool {
    if (!per_field.has_value()) {
      out = scalar;
      return std::isfinite(out) && out >= 0.0;
    }
    if (per_field->size() != count) {
      return false;
    }
    out = (*per_field)[index];
    return std::isfinite(out) && out >= 0.0;
  };

  double atol = 0.0;
  double rtol = 0.0;
  if (!pick(tol.absolute, tol.absolute_per_field, atol) ||
      !pick(tol.relative, tol.relative_per_field, rtol)) {
    return Denominator{DenomKind::NoDecision, 0.0};
  }
  if (rtol == 0.0) {
    if (atol == 0.0) {
      return Denominator{DenomKind::ZeroDenominator, 0.0};
    }
    return Denominator{DenomKind::Ok, atol};
  }

  double scale = 0.0;
  switch (ev.solution_scale) {
  case SolutionScale::Unspecified: return Denominator{DenomKind::NoDecision, 0.0};
  case SolutionScale::Unit: scale = 1.0; break;
  case SolutionScale::Supplied:
    if (!ev.weights.has_value() || ev.weights->size() != count) {
      return Denominator{DenomKind::NoDecision, 0.0};
    }
    scale = (*ev.weights)[index];
    if (!std::isfinite(scale) || scale < 0.0) {
      return Denominator{DenomKind::NoDecision, 0.0};
    }
    break;
  }
  const double den = atol + rtol * scale;
  if (den == 0.0) {
    return Denominator{DenomKind::ZeroDenominator, 0.0};
  }
  return Denominator{DenomKind::Ok, den};
}

[[nodiscard]] inline NormalizedError from_denominator(DenomKind kind, double ratio) {
  if (kind == DenomKind::NoDecision) {
    return make_no_decision();
  }
  if (kind == DenomKind::ZeroDenominator) {
    return NormalizedError{.metric = std::numeric_limits<double>::infinity(),
                           .verdict = StepAttemptVerdict::Reject,
                           .decision_available = true};
  }
  const auto verdict =
      (ratio <= 1.0) ? StepAttemptVerdict::Accept : StepAttemptVerdict::Reject;
  return NormalizedError{
      .metric = ratio, .verdict = verdict, .decision_available = true};
}

} // namespace detail

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

  if (!ev.field_norms.empty()) {
    if (ev.weights.has_value() && ev.weights->size() != ev.field_norms.size()) {
      return detail::make_no_decision();
    }
    double metric = 0.0;
    for (std::size_t i = 0; i < ev.field_norms.size(); ++i) {
      const auto den = detail::field_denominator(ev, tol, i, ev.field_norms.size());
      if (den.kind != detail::DenomKind::Ok) {
        return detail::from_denominator(den.kind, 0.0);
      }
      metric = std::max(metric, ev.field_norms[i] / den.value);
    }
    return detail::from_denominator(detail::DenomKind::Ok, metric);
  }

  const auto den = detail::field_denominator(ev, tol, 0, 1);
  if (den.kind != detail::DenomKind::Ok) {
    return detail::from_denominator(den.kind, 0.0);
  }
  if (!std::isfinite(*ev.combined_metric)) {
    return detail::make_no_decision();
  }
  return detail::from_denominator(detail::DenomKind::Ok,
                                  std::abs(*ev.combined_metric) / den.value);
}

} // namespace pfc::integrator
