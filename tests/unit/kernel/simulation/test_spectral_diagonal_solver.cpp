// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_spectral_diagonal_solver.cpp
 * @brief Catch2 coverage for SpectralDiagonalSolver
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "openpfc/kernel/simulation/spectral_diagonal_solver.hpp"
#include "openpfc/kernel/simulation/solver_contract.hpp"

#include <cmath>
#include <complex>
#include <string>
#include <vector>

using namespace pfc::sim;
using Catch::Matchers::WithinAbs;

namespace {

class MockExecutionService : public ExecutionService {
public:
  std::vector<std::string> last_halos;
  std::vector<std::string> last_boundaries;
  std::vector<double> last_reduce_data;
  std::vector<double> last_reduce_result;
  MPI_Op last_op = MPI_OP_NULL;

  void request_halo_exchange(const std::vector<std::string> &field_names) override {
    last_halos = field_names;
  }

  void prepare_boundaries(const std::vector<std::string> &field_names) override {
    last_boundaries = field_names;
  }

  std::vector<double> global_reduce(const std::vector<double> &data, MPI_Op op) override {
    last_reduce_data = data;
    last_op = op;
    // By default, return a copy of input (serial behavior)
    last_reduce_result = data;
    return last_reduce_result;
  }
};

LinearOperatorDesc make_desc(std::vector<double> diag,
                             std::string id = "spectral_diagonal") {
  return LinearOperatorDesc{std::move(id), std::nullopt, std::move(diag)};
}

LinearOperatorDesc make_complex_desc(std::vector<std::complex<double>> diag,
                                     std::string id = "spectral_diagonal") {
  return LinearOperatorDesc{std::move(id), std::nullopt, std::move(diag)};
}

} // namespace

TEST_CASE("spectral diagonal regular real divide", "[spectral_diagonal][solver]") {
  static_assert(SolveFunction<SpectralDiagonalSolver, std::vector<double>,
                              std::vector<double>>);

  SpectralDiagonalSolver solver;
  std::vector<double> diag{2.0, 4.0, 5.0};
  std::vector<double> rhs{2.0, 8.0, 15.0};
  std::vector<double> target{0.0, 0.0, 0.0};

  MockExecutionService service;
  StageContext ctx{0.0, service};
  SolveOptions opts{};
  opts.absolute_tolerance = 1e-12;

  auto outcome = solver(make_desc(diag), rhs, target, opts, ctx);

  REQUIRE(outcome.status == ConvergenceStatus::converged);
  REQUIRE(outcome.iteration_count == 1);
  REQUIRE_THAT(outcome.final_residual_norm, WithinAbs(0.0, 1e-14));
  REQUIRE(target == std::vector<double>{1.0, 2.0, 3.0});
  REQUIRE(outcome.solution == target);
  REQUIRE(service.last_op == MPI_SUM);
  REQUIRE(service.last_reduce_data.size() == 1);
}

TEST_CASE("spectral diagonal nullspace fail", "[spectral_diagonal][solver]") {
  SpectralDiagonalConfig config;
  config.nullspace_policy = DiagonalNullspacePolicy::fail;
  config.singular_threshold = 1e-14;
  SpectralDiagonalSolver solver(config);

  std::vector<double> diag{2.0, 0.0, 5.0};
  std::vector<double> rhs{2.0, 1.0, 15.0};
  std::vector<double> target{-1.0, -2.0, -3.0};
  const auto target_sentinel = target;

  MockExecutionService service;
  StageContext ctx{0.0, service};
  SolveOptions opts{};

  auto outcome = solver(make_desc(diag), rhs, target, opts, ctx);

  REQUIRE(outcome.status == ConvergenceStatus::ill_conditioned);
  REQUIRE(outcome.failure_cause.has_value());
  REQUIRE(outcome.failure_cause->find("singular mode") != std::string::npos);
  REQUIRE(target == target_sentinel);
}

TEST_CASE("spectral diagonal nullspace project", "[spectral_diagonal][solver]") {
  SpectralDiagonalConfig config;
  config.nullspace_policy = DiagonalNullspacePolicy::project;
  config.singular_threshold = 1e-14;
  config.null_mode_value = 0.0;
  SpectralDiagonalSolver solver(config);

  // Zero mode with compatible RHS (b=0) so residual stays ~0
  std::vector<double> diag{2.0, 0.0, 5.0};
  std::vector<double> rhs{2.0, 0.0, 15.0};
  std::vector<double> target{9.0, 9.0, 9.0};

  MockExecutionService service;
  StageContext ctx{0.0, service};
  SolveOptions opts{};
  opts.absolute_tolerance = 1e-12;

  auto outcome = solver(make_desc(diag), rhs, target, opts, ctx);

  REQUIRE(outcome.status == ConvergenceStatus::converged);
  REQUIRE(outcome.iteration_count == 1);
  REQUIRE_THAT(outcome.final_residual_norm, WithinAbs(0.0, 1e-14));
  REQUIRE(target == std::vector<double>{1.0, 0.0, 3.0});
}

TEST_CASE("spectral diagonal nullspace regularize", "[spectral_diagonal][solver]") {
  SECTION("lambda > 0 produces b/(d+lambda)") {
    SpectralDiagonalConfig config;
    config.nullspace_policy = DiagonalNullspacePolicy::regularize;
    config.regularization = 1.0;
    SpectralDiagonalSolver solver(config);

    std::vector<double> diag{1.0, 2.0, 0.0};
    std::vector<double> rhs{2.0, 6.0, 4.0};
    std::vector<double> target{0.0, 0.0, 0.0};

    MockExecutionService service;
    StageContext ctx{0.0, service};
    SolveOptions opts{};
    // Residual uses original d, so r_i = -b_i * λ/(d_i+λ); set abs_tol accordingly
    opts.absolute_tolerance = 10.0;

    auto outcome = solver(make_desc(diag), rhs, target, opts, ctx);

    REQUIRE(outcome.status == ConvergenceStatus::converged);
    REQUIRE(outcome.iteration_count == 1);
    REQUIRE_THAT(target[0], WithinAbs(2.0 / (1.0 + 1.0), 1e-14));
    REQUIRE_THAT(target[1], WithinAbs(6.0 / (2.0 + 1.0), 1e-14));
    REQUIRE_THAT(target[2], WithinAbs(4.0 / (0.0 + 1.0), 1e-14));
  }

  SECTION("lambda <= 0 rejects without mutating target") {
    SpectralDiagonalConfig config;
    config.nullspace_policy = DiagonalNullspacePolicy::regularize;
    config.regularization = 0.0;
    SpectralDiagonalSolver solver(config);

    std::vector<double> diag{1.0, 2.0};
    std::vector<double> rhs{1.0, 2.0};
    std::vector<double> target{-7.0, -8.0};
    const auto sentinel = target;

    MockExecutionService service;
    StageContext ctx{0.0, service};
    SolveOptions opts{};

    auto outcome = solver(make_desc(diag), rhs, target, opts, ctx);

    REQUIRE(outcome.status == ConvergenceStatus::unknown_failure);
    REQUIRE(outcome.failure_cause.has_value());
    REQUIRE(outcome.failure_cause->find("regularization") != std::string::npos);
    REQUIRE(target == sentinel);
  }
}

TEST_CASE("spectral diagonal residual threshold", "[spectral_diagonal][solver]") {
  SpectralDiagonalConfig config;
  config.nullspace_policy = DiagonalNullspacePolicy::project;
  config.null_mode_value = 0.0;
  SpectralDiagonalSolver solver(config);

  // Singular mode with incompatible RHS → |r| = |b| on that mode
  std::vector<double> diag{2.0, 0.0};
  std::vector<double> rhs{2.0, 5.0};
  std::vector<double> target{-1.0, -1.0};
  const auto sentinel = target;

  MockExecutionService service;
  StageContext ctx{0.0, service};
  SolveOptions opts{};
  opts.absolute_tolerance = 1e-12;

  auto outcome = solver(make_desc(diag), rhs, target, opts, ctx);

  REQUIRE(outcome.status != ConvergenceStatus::converged);
  REQUIRE(outcome.final_residual_norm > *opts.absolute_tolerance);
  REQUIRE_THAT(outcome.final_residual_norm, WithinAbs(5.0, 1e-12));
  REQUIRE(target == sentinel);
}

TEST_CASE("spectral diagonal inputs unchanged", "[spectral_diagonal][solver]") {
  SpectralDiagonalSolver solver;
  std::vector<double> diag{2.0, 4.0, 5.0};
  std::vector<double> rhs{2.0, 8.0, 15.0};
  const auto diag_copy = diag;
  const auto rhs_copy = rhs;
  std::vector<double> target{0.0, 0.0, 0.0};

  LinearOperatorDesc desc = make_desc(diag);
  MockExecutionService service;
  StageContext ctx{0.0, service};
  SolveOptions opts{};
  opts.absolute_tolerance = 1e-12;

  auto outcome = solver(desc, rhs, target, opts, ctx);

  REQUIRE(outcome.status == ConvergenceStatus::converged);
  REQUIRE(rhs == rhs_copy);
  REQUIRE(std::get<std::vector<double>>(desc.operator_context) == diag_copy);
}

TEST_CASE("spectral diagonal empty operator_identifier accepted",
          "[spectral_diagonal][solver]") {
  SpectralDiagonalSolver solver;
  std::vector<double> diag{3.0};
  std::vector<double> rhs{6.0};
  std::vector<double> target{0.0};

  MockExecutionService service;
  StageContext ctx{0.0, service};
  SolveOptions opts{};
  opts.absolute_tolerance = 1e-12;

  auto outcome = solver(make_desc(diag, ""), rhs, target, opts, ctx);
  REQUIRE(outcome.status == ConvergenceStatus::converged);
  REQUIRE_THAT(target[0], WithinAbs(2.0, 1e-14));
}

TEST_CASE("spectral diagonal complex regular divide",
          "[spectral_diagonal][solver]") {
  using Complex = std::complex<double>;
  static_assert(SolveFunction<SpectralDiagonalSolver, std::vector<Complex>,
                              std::vector<Complex>>);

  SpectralDiagonalSolver solver;
  std::vector<Complex> diag{Complex{2.0, 0.0}, Complex{4.0, 0.0},
                            Complex{5.0, 0.0}};
  std::vector<Complex> rhs{Complex{2.0, 0.0}, Complex{8.0, 0.0},
                           Complex{15.0, 0.0}};
  std::vector<Complex> target{Complex{0.0, 0.0}, Complex{0.0, 0.0},
                              Complex{0.0, 0.0}};

  MockExecutionService service;
  StageContext ctx{0.0, service};
  SolveOptions opts{};
  opts.absolute_tolerance = 1e-12;

  auto outcome = solver(make_complex_desc(diag), rhs, target, opts, ctx);

  REQUIRE(outcome.status == ConvergenceStatus::converged);
  REQUIRE(outcome.iteration_count == 1);
  REQUIRE_THAT(outcome.final_residual_norm, WithinAbs(0.0, 1e-14));
  REQUIRE(target ==
          std::vector<Complex>{Complex{1.0, 0.0}, Complex{2.0, 0.0},
                               Complex{3.0, 0.0}});
  REQUIRE(outcome.solution == target);
  REQUIRE(service.last_op == MPI_SUM);
  REQUIRE(service.last_reduce_data.size() == 1);
}

TEST_CASE("spectral diagonal complex nullspace fail",
          "[spectral_diagonal][solver]") {
  using Complex = std::complex<double>;
  SpectralDiagonalConfig config;
  config.nullspace_policy = DiagonalNullspacePolicy::fail;
  config.singular_threshold = 1e-14;
  SpectralDiagonalSolver solver(config);

  std::vector<Complex> diag{Complex{2.0, 0.0}, Complex{0.0, 0.0},
                            Complex{5.0, 0.0}};
  std::vector<Complex> rhs{Complex{2.0, 0.0}, Complex{1.0, 0.0},
                           Complex{15.0, 0.0}};
  std::vector<Complex> target{Complex{-1.0, 0.0}, Complex{-2.0, 0.0},
                              Complex{-3.0, 0.0}};
  const auto target_sentinel = target;

  MockExecutionService service;
  StageContext ctx{0.0, service};
  SolveOptions opts{};

  auto outcome = solver(make_complex_desc(diag), rhs, target, opts, ctx);

  REQUIRE(outcome.status == ConvergenceStatus::ill_conditioned);
  REQUIRE(outcome.failure_cause.has_value());
  REQUIRE(outcome.failure_cause->find("singular mode") != std::string::npos);
  REQUIRE(target == target_sentinel);
}

TEST_CASE("spectral diagonal complex nullspace project",
          "[spectral_diagonal][solver]") {
  using Complex = std::complex<double>;
  SpectralDiagonalConfig config;
  config.nullspace_policy = DiagonalNullspacePolicy::project;
  config.singular_threshold = 1e-14;
  config.null_mode_value = 0.0;
  SpectralDiagonalSolver solver(config);

  // Zero mode with compatible RHS (b=0) so residual stays ~0
  std::vector<Complex> diag{Complex{2.0, 0.0}, Complex{0.0, 0.0},
                            Complex{5.0, 0.0}};
  std::vector<Complex> rhs{Complex{2.0, 0.0}, Complex{0.0, 0.0},
                           Complex{15.0, 0.0}};
  std::vector<Complex> target{Complex{9.0, 0.0}, Complex{9.0, 0.0},
                              Complex{9.0, 0.0}};

  MockExecutionService service;
  StageContext ctx{0.0, service};
  SolveOptions opts{};
  opts.absolute_tolerance = 1e-12;

  auto outcome = solver(make_complex_desc(diag), rhs, target, opts, ctx);

  REQUIRE(outcome.status == ConvergenceStatus::converged);
  REQUIRE(outcome.iteration_count == 1);
  REQUIRE_THAT(outcome.final_residual_norm, WithinAbs(0.0, 1e-14));
  REQUIRE(target ==
          std::vector<Complex>{Complex{1.0, 0.0}, Complex{0.0, 0.0},
                               Complex{3.0, 0.0}});
}

TEST_CASE("spectral diagonal complex nullspace regularize",
          "[spectral_diagonal][solver]") {
  using Complex = std::complex<double>;
  SpectralDiagonalConfig config;
  config.nullspace_policy = DiagonalNullspacePolicy::regularize;
  config.regularization = 1.0;
  SpectralDiagonalSolver solver(config);

  std::vector<Complex> diag{Complex{1.0, 0.0}, Complex{2.0, 0.0},
                            Complex{0.0, 0.0}};
  std::vector<Complex> rhs{Complex{2.0, 0.0}, Complex{6.0, 0.0},
                           Complex{4.0, 0.0}};
  std::vector<Complex> target{Complex{0.0, 0.0}, Complex{0.0, 0.0},
                              Complex{0.0, 0.0}};

  MockExecutionService service;
  StageContext ctx{0.0, service};
  SolveOptions opts{};
  // Residual uses original d, so r_i = -b_i * λ/(d_i+λ); set abs_tol accordingly
  opts.absolute_tolerance = 10.0;

  auto outcome = solver(make_complex_desc(diag), rhs, target, opts, ctx);

  REQUIRE(outcome.status == ConvergenceStatus::converged);
  REQUIRE(outcome.iteration_count == 1);
  REQUIRE_THAT(target[0].real(), WithinAbs(2.0 / (1.0 + 1.0), 1e-14));
  REQUIRE_THAT(target[0].imag(), WithinAbs(0.0, 1e-14));
  REQUIRE_THAT(target[1].real(), WithinAbs(6.0 / (2.0 + 1.0), 1e-14));
  REQUIRE_THAT(target[1].imag(), WithinAbs(0.0, 1e-14));
  REQUIRE_THAT(target[2].real(), WithinAbs(4.0 / (0.0 + 1.0), 1e-14));
  REQUIRE_THAT(target[2].imag(), WithinAbs(0.0, 1e-14));
}

TEST_CASE("spectral diagonal complex residual threshold",
          "[spectral_diagonal][solver]") {
  using Complex = std::complex<double>;
  SpectralDiagonalConfig config;
  config.nullspace_policy = DiagonalNullspacePolicy::project;
  config.null_mode_value = 0.0;
  SpectralDiagonalSolver solver(config);

  // Singular mode with incompatible RHS → |r| = |b| on that mode
  std::vector<Complex> diag{Complex{2.0, 0.0}, Complex{0.0, 0.0}};
  std::vector<Complex> rhs{Complex{2.0, 0.0}, Complex{5.0, 0.0}};
  std::vector<Complex> target{Complex{-1.0, 0.0}, Complex{-1.0, 0.0}};
  const auto sentinel = target;

  MockExecutionService service;
  StageContext ctx{0.0, service};
  SolveOptions opts{};
  opts.absolute_tolerance = 1e-12;

  auto outcome = solver(make_complex_desc(diag), rhs, target, opts, ctx);

  REQUIRE(outcome.status != ConvergenceStatus::converged);
  REQUIRE(outcome.final_residual_norm > *opts.absolute_tolerance);
  REQUIRE_THAT(outcome.final_residual_norm, WithinAbs(5.0, 1e-12));
  REQUIRE(target == sentinel);
}

TEST_CASE("spectral diagonal complex inputs unchanged",
          "[spectral_diagonal][solver]") {
  using Complex = std::complex<double>;
  SpectralDiagonalSolver solver;
  std::vector<Complex> diag{Complex{2.0, 0.0}, Complex{4.0, 0.0},
                            Complex{5.0, 0.0}};
  std::vector<Complex> rhs{Complex{2.0, 0.0}, Complex{8.0, 0.0},
                           Complex{15.0, 0.0}};
  const auto diag_copy = diag;
  const auto rhs_copy = rhs;
  std::vector<Complex> target{Complex{0.0, 0.0}, Complex{0.0, 0.0},
                              Complex{0.0, 0.0}};

  LinearOperatorDesc desc = make_complex_desc(diag);
  MockExecutionService service;
  StageContext ctx{0.0, service};
  SolveOptions opts{};
  opts.absolute_tolerance = 1e-12;

  auto outcome = solver(desc, rhs, target, opts, ctx);

  REQUIRE(outcome.status == ConvergenceStatus::converged);
  REQUIRE(rhs == rhs_copy);
  REQUIRE(std::get<std::vector<Complex>>(desc.operator_context) == diag_copy);
}

TEST_CASE("spectral diagonal injected reduced sum affects convergence",
          "[spectral_diagonal][solver]") {
  SpectralDiagonalConfig config;
  config.nullspace_policy = DiagonalNullspacePolicy::project;
  config.null_mode_value = 0.0;
  SpectralDiagonalSolver solver(config);

  // Singular mode with incompatible RHS → local sum_sq = 25.0 (r = -b = -5.0)
  std::vector<double> diag{2.0, 0.0};
  std::vector<double> rhs{2.0, 5.0};
  std::vector<double> target{-1.0, -1.0};
  const auto sentinel = target;

  // Mock simulates a 2-rank MPI run with each rank having sum_sq = 25.0
  // The reduced sum would be 50.0, so residual_norm = sqrt(50.0) ~ 7.07
  class ReducingMockExecutionService : public ExecutionService {
  public:
    std::vector<double> last_reduce_data;
    MPI_Op last_op = MPI_OP_NULL;

    void request_halo_exchange(const std::vector<std::string> &) override {}
    void prepare_boundaries(const std::vector<std::string> &) override {}

    std::vector<double> global_reduce(const std::vector<double> &data, MPI_Op op) override {
      last_reduce_data = data;
      last_op = op;
      // Simulate 2-rank MPI: input is local sum from one rank, return global sum
      // If local sum_sq is 25.0, return 50.0 (sum from 2 ranks)
      return {data[0] * 2.0};
    }
  };

  ReducingMockExecutionService service;
  StageContext ctx{0.0, service};
  SolveOptions opts{};
  opts.absolute_tolerance = 1e-12;

  auto outcome = solver(make_desc(diag), rhs, target, opts, ctx);

  // Should fail because the mock injected a large reduced sum (50.0)
  // residual_norm = sqrt(50.0) ~ 7.07, which exceeds tolerance 1e-12
  REQUIRE(outcome.status != ConvergenceStatus::converged);
  REQUIRE(outcome.status == ConvergenceStatus::max_iterations_reached);
  // Verify the reduced sum was actually used: sqrt(50.0) ≈ 7.07
  REQUIRE_THAT(outcome.final_residual_norm, WithinAbs(std::sqrt(50.0), 1e-10));
  REQUIRE(service.last_op == MPI_SUM);
  REQUIRE(service.last_reduce_data.size() == 1);
  // Local sum_sq from one rank is 25.0 (r = -5.0 on singular mode)
  REQUIRE_THAT(service.last_reduce_data[0], WithinAbs(25.0, 1e-10));
  REQUIRE(target == sentinel);  // Target unchanged on convergence failure
}

