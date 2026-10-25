// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/integrator/error_evidence.hpp>

#include <cmath>
#include <vector>

#include <mpi.h>

using namespace pfc::integrator;
using Catch::Matchers::WithinAbs;

TEST_CASE("embedded_pair_scalar_normalize_accept", "[error_evidence]") {
  const double norms[] = {0.01};
  auto ev = make_embedded_pair_evidence(norms, AggregationScope::AlreadyReduced,
                                        /*order_tag=*/3);
  REQUIRE(ev.valid);
  REQUIRE(ev.kind == EvidenceKind::EmbeddedPair);
  REQUIRE(ev.field_norms.size() == 1);
  REQUIRE(ev.order_tag == 3);

  const ErrorTolerances tol{.absolute = 1e-2, .relative = 0.0};
  const auto n = normalize_error_evidence(ev, tol);
  REQUIRE(n.decision_available);
  REQUIRE(n.verdict == StepAttemptVerdict::Accept);
  REQUIRE_THAT(n.metric, WithinAbs(1.0, 1e-15)); // 0.01 / 0.01
}

TEST_CASE("embedded_pair_scalar_normalize_reject", "[error_evidence]") {
  const double norms[] = {0.05};
  auto ev = make_embedded_pair_evidence(norms, AggregationScope::AlreadyReduced);
  const ErrorTolerances tol{.absolute = 1e-2, .relative = 0.0};
  const auto n = normalize_error_evidence(ev, tol);
  REQUIRE(n.decision_available);
  REQUIRE(n.verdict == StepAttemptVerdict::Reject);
  REQUIRE_THAT(n.metric, WithinAbs(5.0, 1e-15)); // 0.05 / 0.01
}

TEST_CASE("residual_multifield_normalize", "[error_evidence]") {
  const double norms[] = {1e-4, 2e-4};
  const double weights[] = {1.0, 2.0};
  auto ev =
      make_residual_evidence(norms, AggregationScope::AlreadyReduced,
                             /*order_tag=*/{}, std::span<const double>{weights});
  REQUIRE(ev.valid);
  REQUIRE(ev.kind == EvidenceKind::ResidualAPosteriori);
  REQUIRE(ev.field_norms.size() == 2);

  // den_0 = 1e-3 + 0 = 1e-3 → e0 = 0.1; den_1 = 1e-3 + 0*2 → e1 = 0.2
  const ErrorTolerances tol{.absolute = 1e-3, .relative = 0.0};
  const auto n = normalize_error_evidence(ev, tol);
  REQUIRE(n.decision_available);
  REQUIRE(n.verdict == StepAttemptVerdict::Accept);
  REQUIRE_THAT(n.metric, WithinAbs(0.2, 1e-15));

  // Same normalize path as EmbeddedPair — no kind-specific controller code.
  auto embedded = make_embedded_pair_evidence(
      norms, AggregationScope::AlreadyReduced, {}, std::span<const double>{weights});
  const auto n2 = normalize_error_evidence(embedded, tol);
  REQUIRE_THAT(n2.metric, WithinAbs(n.metric, 1e-15));
  REQUIRE(n2.verdict == n.verdict);
}

TEST_CASE("invalid_evidence_yields_no_decision", "[error_evidence]") {
  auto ev = make_invalid_evidence(EvidenceKind::EmbeddedPair);
  REQUIRE_FALSE(ev.valid);
  REQUIRE(ev.field_norms.empty());

  const ErrorTolerances tol{.absolute = 1e-6, .relative = 1e-3};
  const auto n = normalize_error_evidence(ev, tol);
  REQUIRE_FALSE(n.decision_available);
  REQUIRE(n.verdict == StepAttemptVerdict::NoDecision);
  REQUIRE(std::isnan(n.metric));
}

TEST_CASE("reduce_rank_local_single_rank_identity", "[error_evidence]") {
  const double norms[] = {0.1, 0.2, 0.3};
  auto ev = make_embedded_pair_evidence(norms, AggregationScope::RankLocal);
  REQUIRE(ev.scope == AggregationScope::RankLocal);
  const auto before = ev.field_norms;

  auto reduced = reduce_error_evidence(ev, MPI_COMM_WORLD);
  REQUIRE(reduced.valid);
  REQUIRE(reduced.scope == AggregationScope::AlreadyReduced);
  REQUIRE(reduced.field_norms.size() == before.size());
  for (std::size_t i = 0; i < before.size(); ++i) {
    REQUIRE(reduced.field_norms[i] == before[i]); // bitwise identity
  }
}

TEST_CASE("already_reduced_not_double_reduced", "[error_evidence]") {
  const double norms[] = {0.7, 0.8};
  auto ev = make_residual_evidence(norms, AggregationScope::AlreadyReduced);
  REQUIRE(ev.scope == AggregationScope::AlreadyReduced);

  auto once = reduce_error_evidence(ev, MPI_COMM_WORLD);
  REQUIRE(once.scope == AggregationScope::AlreadyReduced);
  REQUIRE(once.field_norms == ev.field_norms);

  auto twice = reduce_error_evidence(once, MPI_COMM_WORLD);
  REQUIRE(twice.scope == AggregationScope::AlreadyReduced);
  REQUIRE(twice.field_norms == once.field_norms);
  REQUIRE(twice.field_norms == ev.field_norms);
}

TEST_CASE("method_specific_extension_hook_constructible", "[error_evidence]") {
  const double norms[] = {1e-5, 2e-5};
  auto ev = make_method_specific_evidence(norms, AggregationScope::RankLocal,
                                          /*order_tag=*/4);
  REQUIRE(ev.valid);
  REQUIRE(ev.kind == EvidenceKind::MethodSpecific);
  REQUIRE(ev.order_tag == 4);
  REQUIRE(ev.field_norms.size() == 2);

  // Still uses the shared normalize path (no method-specific controller).
  auto reduced = reduce_error_evidence(ev);
  const ErrorTolerances tol{.absolute = 1e-4, .relative = 0.0};
  const auto n = normalize_error_evidence(reduced, tol);
  REQUIRE(n.decision_available);
  REQUIRE(n.verdict == StepAttemptVerdict::Accept);
}

TEST_CASE("negative_or_empty_norms_yield_invalid_factory", "[error_evidence]") {
  const double bad[] = {-1.0};
  auto ev = make_embedded_pair_evidence(bad, AggregationScope::RankLocal);
  REQUIRE_FALSE(ev.valid);

  auto empty = make_residual_evidence({}, AggregationScope::RankLocal);
  REQUIRE_FALSE(empty.valid);
}

TEST_CASE("pure atol does not need a solution scale", "[error_evidence]") {
  const double norms[] = {2e-3};
  auto ev = make_embedded_pair_evidence(norms, AggregationScope::AlreadyReduced);
  REQUIRE(ev.solution_scale == SolutionScale::Unspecified);
  const ErrorTolerances tol{.absolute = 1e-3, .relative = 0.0};
  const auto n = normalize_error_evidence(ev, tol);
  REQUIRE(n.verdict == StepAttemptVerdict::Reject);
  REQUIRE_THAT(n.metric, WithinAbs(2.0, 1e-15));
}

TEST_CASE("pure rtol uses the supplied scale", "[error_evidence]") {
  const double norms[] = {0.02};
  const double scale[] = {4.0};
  auto ev = make_embedded_pair_evidence(norms, AggregationScope::AlreadyReduced, {},
                                        std::span<const double>{scale});
  REQUIRE(ev.solution_scale == SolutionScale::Supplied);
  const ErrorTolerances tol{.absolute = 0.0, .relative = 1e-2};
  const auto n = normalize_error_evidence(ev, tol);
  // den = 1e-2 * 4 = 0.04, metric = 0.5.
  REQUIRE(n.verdict == StepAttemptVerdict::Accept);
  REQUIRE_THAT(n.metric, WithinAbs(0.5, 1e-15));
}

TEST_CASE("mixed atol and rtol", "[error_evidence]") {
  const double norms[] = {0.0101};
  const double scale[] = {1.0};
  auto ev = make_embedded_pair_evidence(norms, AggregationScope::AlreadyReduced, {},
                                        std::span<const double>{scale});
  const ErrorTolerances tol{.absolute = 1e-4, .relative = 1e-2};
  const auto n = normalize_error_evidence(ev, tol);
  REQUIRE_THAT(n.metric, WithinAbs(1.0, 1e-12));
  REQUIRE(n.verdict == StepAttemptVerdict::Accept);
}

TEST_CASE("unit scale is explicit and unspecified rtol fails closed",
          "[error_evidence]") {
  const double norms[] = {2e-5};
  auto unit = make_embedded_pair_evidence(norms, AggregationScope::AlreadyReduced,
                                          {}, {}, SolutionScale::Unit);
  const ErrorTolerances tol{.absolute = 1e-5, .relative = 1e-5};
  const auto n = normalize_error_evidence(unit, tol);
  REQUIRE_THAT(n.metric, WithinAbs(1.0, 1e-12));

  auto missing =
      make_embedded_pair_evidence(norms, AggregationScope::AlreadyReduced);
  const auto closed = normalize_error_evidence(missing, tol);
  REQUIRE_FALSE(closed.decision_available);
  REQUIRE(closed.verdict == StepAttemptVerdict::NoDecision);
}

TEST_CASE("zero scale rejects a pure relative tolerance", "[error_evidence]") {
  const double norms[] = {1e-8};
  const double scale[] = {0.0};
  auto ev = make_embedded_pair_evidence(norms, AggregationScope::AlreadyReduced, {},
                                        std::span<const double>{scale});
  const ErrorTolerances pure{.absolute = 0.0, .relative = 1e-4};
  const auto rejected = normalize_error_evidence(ev, pure);
  REQUIRE(rejected.decision_available);
  REQUIRE(rejected.verdict == StepAttemptVerdict::Reject);
  REQUIRE(std::isinf(rejected.metric));

  const ErrorTolerances mixed{.absolute = 1e-6, .relative = 1e-4};
  const auto accepted = normalize_error_evidence(ev, mixed);
  REQUIRE(accepted.verdict == StepAttemptVerdict::Accept);
  REQUIRE_THAT(accepted.metric, WithinAbs(0.01, 1e-15));
}

TEST_CASE("near-zero scale inflates a relative metric", "[error_evidence]") {
  const double norms[] = {1e-8};
  const double scale[] = {1e-16};
  auto ev = make_embedded_pair_evidence(norms, AggregationScope::AlreadyReduced, {},
                                        std::span<const double>{scale});
  const ErrorTolerances tol{.absolute = 0.0, .relative = 1e-6};
  const auto n = normalize_error_evidence(ev, tol);
  REQUIRE(n.verdict == StepAttemptVerdict::Reject);
  REQUIRE_THAT(n.metric, WithinAbs(1e14, 1e2));
}

TEST_CASE("several fields and per-field tolerances", "[error_evidence]") {
  const double norms[] = {1e-4, 2e-3};
  const double scales[] = {1.0, 10.0};
  auto ev = make_embedded_pair_evidence(norms, AggregationScope::AlreadyReduced, {},
                                        std::span<const double>{scales});
  const ErrorTolerances tol{.absolute = 0.0, .relative = 1e-4};
  const auto n = normalize_error_evidence(ev, tol);
  REQUIRE_THAT(n.metric, WithinAbs(2.0, 1e-12));

  ErrorTolerances per_field;
  per_field.absolute_per_field = std::vector<double>{1.0, 1e-3};
  per_field.relative_per_field = std::vector<double>{0.0, 0.0};
  const double equal[] = {0.1, 0.1};
  const double ones[] = {1.0, 1.0};
  auto fields = make_residual_evidence(equal, AggregationScope::AlreadyReduced, {},
                                       std::span<const double>{ones});
  const auto scaled = normalize_error_evidence(fields, per_field);
  REQUIRE_THAT(scaled.metric, WithinAbs(100.0, 1e-12));
}

TEST_CASE("weights override a requested unit scale", "[error_evidence]") {
  const double norms[] = {0.2};
  const double scale[] = {2.0};
  auto ev = make_embedded_pair_evidence(norms, AggregationScope::AlreadyReduced, {},
                                        std::span<const double>{scale},
                                        SolutionScale::Unit);
  REQUIRE(ev.solution_scale == SolutionScale::Supplied);
  const ErrorTolerances tol{.absolute = 0.0, .relative = 0.1};
  const auto n = normalize_error_evidence(ev, tol);
  REQUIRE_THAT(n.metric, WithinAbs(1.0, 1e-12));
}

TEST_CASE("reduce maxes norms and scales or fails closed", "[error_evidence][MPI]") {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const double norm = (size > 1 && rank != 0) ? 0.2 : 1.0;
  const double scale = (size > 1 && rank != 0) ? 0.1 : 10.0;
  const double norms[] = {norm};
  const double scales[] = {scale};
  auto ev = make_embedded_pair_evidence(norms, AggregationScope::RankLocal, {},
                                        std::span<const double>{scales});
  auto reduced = reduce_error_evidence(ev);
  REQUIRE(reduced.valid);
  REQUIRE(reduced.scope == AggregationScope::AlreadyReduced);
  if (size < 2) {
    REQUIRE(reduced.field_norms[0] == 1.0);
    REQUIRE((*reduced.weights)[0] == 10.0);
    return;
  }
  // ||e||_inf / ||y||_inf = 1/10. The max of the local ratios would be 2.
  REQUIRE(reduced.field_norms[0] == 1.0);
  REQUIRE((*reduced.weights)[0] == 10.0);
  const ErrorTolerances tol{.absolute = 0.0, .relative = 1.0};
  const auto n = normalize_error_evidence(reduced, tol);
  REQUIRE_THAT(n.metric, WithinAbs(0.1, 1e-15));

  const double one[] = {0.1};
  auto unit = make_embedded_pair_evidence(one, AggregationScope::RankLocal, {}, {},
                                          SolutionScale::Unit);
  ErrorEvidence supplied;
  if (rank == 0) {
    supplied = unit;
  } else {
    const double w[] = {1.0};
    supplied = make_embedded_pair_evidence(one, AggregationScope::RankLocal, {},
                                           std::span<const double>{w});
  }
  auto disagreed = reduce_error_evidence(supplied);
  REQUIRE_FALSE(disagreed.valid);

  const double first[] = {0.1};
  const double second[] = {0.1, 0.2};
  auto shaped =
      (rank == 0) ? make_embedded_pair_evidence(first, AggregationScope::RankLocal)
                  : make_embedded_pair_evidence(second, AggregationScope::RankLocal);
  auto bad_shape = reduce_error_evidence(shaped);
  REQUIRE_FALSE(bad_shape.valid);

  // One invalid rank must not leave the collective. The other ranks fail
  // closed with it, before any field buffer is reduced.
  ErrorEvidence mixed =
      (rank == 0) ? make_invalid_evidence(EvidenceKind::EmbeddedPair)
                  : make_embedded_pair_evidence(one, AggregationScope::RankLocal);
  auto bad_valid = reduce_error_evidence(mixed);
  REQUIRE_FALSE(bad_valid.valid);
  REQUIRE(bad_valid.scope == AggregationScope::AlreadyReduced);
}
