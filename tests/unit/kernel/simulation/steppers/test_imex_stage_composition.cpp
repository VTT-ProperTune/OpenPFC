// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <mpi.h>
#include <string>
#include <vector>

#include <openpfc/kernel/simulation/solver_contract.hpp>
#include <openpfc/kernel/simulation/steppers/imex_stage_composition.hpp>

using pfc::sim::ConvergenceStatus;
using pfc::sim::ExecutionService;
using pfc::sim::LinearOperatorDesc;
using pfc::sim::SolveOptions;
using pfc::sim::SolveOutcome;
using pfc::sim::StageContext;
using pfc::sim::steppers::ExplicitOperatorEval;
using pfc::sim::steppers::ImexEulerComposer;

namespace {

class MockExecutionService : public ExecutionService {
public:
  void request_halo_exchange(const std::vector<std::string> &) override {}
  void prepare_boundaries(const std::vector<std::string> &) override {}
  std::vector<double> global_reduce(const std::vector<double> &data, MPI_Op) override {
    // Return copy of input (serial behavior)
    return data;
  }
};

struct CountingExplicitEval {
  int *count{nullptr};
  void operator()(double /*t*/, const std::vector<double> &u,
                  std::vector<double> &du) const {
    if (count != nullptr) {
      ++(*count);
    }
    for (std::size_t i = 0; i < u.size(); ++i) {
      du[i] = -u[i];
    }
  }
};

struct IdentitySuccessSolver {
  int *explicit_count_at_entry{nullptr};
  int *required_explicit_count{nullptr};

  SolveOutcome<std::vector<double> &>
  operator()(const LinearOperatorDesc &, const std::vector<double> &rhs,
             std::vector<double> &target_out, const SolveOptions &,
             const StageContext &) const {
    if (explicit_count_at_entry != nullptr &&
        required_explicit_count != nullptr) {
      *explicit_count_at_entry = *required_explicit_count;
    }
    target_out = rhs;
    return SolveOutcome<std::vector<double> &>{
        target_out, ConvergenceStatus::converged, 1, 0.0, std::nullopt};
  }
};

struct IllConditionedSolver {
  SolveOutcome<std::vector<double> &>
  operator()(const LinearOperatorDesc &, const std::vector<double> &,
             std::vector<double> & /*target_out*/, const SolveOptions &,
             const StageContext &) const {
    // Intentionally do not write target_out.
    static std::vector<double> unused;
    return SolveOutcome<std::vector<double> &>{
        unused, ConvergenceStatus::ill_conditioned, 0, 1.0,
        std::string("ill-conditioned fixture")};
  }
};

} // namespace

TEST_CASE("candidate_isolation_until_apply",
          "[imex_stage_composition][unit]") {
  static_assert(ExplicitOperatorEval<CountingExplicitEval>);

  constexpr std::size_t n = 3;
  int explicit_count = 0;
  CountingExplicitEval eval{&explicit_count};
  IdentitySuccessSolver solver;
  ImexEulerComposer composer(n, eval, solver);
  static_assert(
      pfc::sim::SolveFunction<IdentitySuccessSolver, std::vector<double>,
                              std::vector<double>>);

  std::vector<double> u{1.0, -2.0, 0.5};
  const std::vector<double> fingerprint = u;

  MockExecutionService exec;
  StageContext ctx{0.0, exec};
  SolveOptions opts;
  LinearOperatorDesc op{"identity_proof"};

  const double t = 1.25;
  const double dt = 0.1;
  auto result = composer.attempt(t, dt, u, op, opts, ctx);

  REQUIRE(result.success);
  REQUIRE(result.t0 == t);
  REQUIRE(result.dt == dt);
  REQUIRE(result.t1 == t + dt);
  REQUIRE(result.solve_status == ConvergenceStatus::converged);
  REQUIRE(result.solve_iterations == 1);
  REQUIRE(result.final_residual_norm == 0.0);
  REQUIRE(u == fingerprint);
  REQUIRE(result.candidate.size() == n);

  // Inspecting candidate must not mutate accepted state.
  (void)result.candidate[0];
  REQUIRE(u == fingerprint);

  composer.apply_candidate(u);
  REQUIRE(u != fingerprint);
  REQUIRE(u == result.candidate);
}

TEST_CASE("failure_does_not_mutate_accepted",
          "[imex_stage_composition][unit]") {
  constexpr std::size_t n = 2;
  CountingExplicitEval eval{nullptr};
  IllConditionedSolver solver;
  ImexEulerComposer composer(n, eval, solver);

  std::vector<double> u{3.0, -1.5};
  const std::vector<double> fingerprint = u;

  MockExecutionService exec;
  StageContext ctx{0.0, exec};
  auto result =
      composer.attempt(0.0, 0.25, u, LinearOperatorDesc{"fail"}, SolveOptions{},
                       ctx);

  REQUIRE_FALSE(result.success);
  REQUIRE(result.solve_status == ConvergenceStatus::ill_conditioned);
  REQUIRE(result.failure_cause.has_value());
  REQUIRE(u == fingerprint);
  REQUIRE(result.t1 == result.t0);
}

TEST_CASE("explicit_then_implicit_ordering",
          "[imex_stage_composition][unit]") {
  constexpr std::size_t n = 1;
  int explicit_count = 0;
  int explicit_seen_in_solver = -1;
  CountingExplicitEval eval{&explicit_count};
  IdentitySuccessSolver solver{&explicit_seen_in_solver, &explicit_count};
  ImexEulerComposer composer(n, eval, solver);

  std::vector<double> u{2.0};
  MockExecutionService exec;
  StageContext ctx{-1.0, exec};

  auto result = composer.attempt(0.5, 0.1, u, LinearOperatorDesc{"order"},
                                 SolveOptions{}, ctx);

  REQUIRE(result.success);
  REQUIRE(explicit_count == 1);
  REQUIRE(explicit_seen_in_solver == 1);
  REQUIRE(ctx.evaluation_time == 0.5);
}
