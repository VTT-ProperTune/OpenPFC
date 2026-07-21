// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/integrator/error_evidence.hpp>

#include <cmath>
#include <vector>

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
  auto ev =
      make_embedded_pair_evidence(norms, AggregationScope::AlreadyReduced);
  const ErrorTolerances tol{.absolute = 1e-2, .relative = 0.0};
  const auto n = normalize_error_evidence(ev, tol);
  REQUIRE(n.decision_available);
  REQUIRE(n.verdict == StepAttemptVerdict::Reject);
  REQUIRE_THAT(n.metric, WithinAbs(5.0, 1e-15)); // 0.05 / 0.01
}

TEST_CASE("residual_multifield_normalize", "[error_evidence]") {
  const double norms[] = {1e-4, 2e-4};
  const double weights[] = {1.0, 2.0};
  auto ev = make_residual_evidence(norms, AggregationScope::AlreadyReduced,
                                   /*order_tag=*/{},
                                   std::span<const double>{weights});
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
      norms, AggregationScope::AlreadyReduced, {},
      std::span<const double>{weights});
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
  auto ev =
      make_residual_evidence(norms, AggregationScope::AlreadyReduced);
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
