// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/simulation/steppers/euler.hpp>
#include <openpfc/kernel/simulation/steppers/imex_euler.hpp>

#include <array>
#include <cmath>
#include <tuple>
#include <vector>

using namespace pfc::sim;
using namespace pfc::sim::steppers;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

class MockExecutionService : public ExecutionService {
public:
  void request_halo_exchange(const std::vector<std::string> &) override {}
  void prepare_boundaries(const std::vector<std::string> &) override {}
  void global_reduce(const std::vector<double> &, MPI_Op) override {}
};

struct ConstantRHS {
  double c;
  void operator()(double /*t*/, std::vector<double> & /*u*/,
                  std::vector<double> &du) const {
    for (double &v : du) {
      v = c;
    }
  }
};

struct ZeroRHS {
  void operator()(double /*t*/, std::vector<double> & /*u*/,
                  std::vector<double> &du) const {
    for (double &v : du) {
      v = 0.0;
    }
  }
};

struct CompositeConstantRHS {
  double c_field1;
  double c_field2;
  void operator()(double /*t*/,
                  std::tuple<std::vector<double> &, std::vector<double> &> /*u*/,
                  std::tuple<std::vector<double> &, std::vector<double> &> du)
      const {
    auto &du1 = std::get<0>(du);
    auto &du2 = std::get<1>(du);
    for (std::size_t i = 0; i < du1.size(); ++i) {
      du1[i] = c_field1;
      du2[i] = c_field2;
    }
  }
};

auto make_identity_solver() {
  return [](const LinearOperatorDesc &, const auto &rhs, auto &target,
            const SolveOptions &,
            const StageContext &) -> SolveOutcome<std::decay_t<decltype(target)>> {
    using TargetType = std::decay_t<decltype(target)>;
    constexpr std::size_t n = std::tuple_size_v<TargetType>;
    auto copy_fields = [&]<std::size_t... I>(std::index_sequence<I...>) {
      ((std::get<I>(target) = std::get<I>(rhs)), ...);
    };
    copy_fields(std::make_index_sequence<n>{});
    return SolveOutcome<TargetType>{target, ConvergenceStatus::converged, 1,
                                    0.0, std::nullopt};
  };
}

auto make_failing_solver() {
  return [](const LinearOperatorDesc &, const auto & /*rhs*/, auto &target,
            const SolveOptions &,
            const StageContext &) -> SolveOutcome<std::decay_t<decltype(target)>> {
    using TargetType = std::decay_t<decltype(target)>;
    return SolveOutcome<TargetType>{
        target, ConvergenceStatus::unknown_failure, 0, 1.0,
        std::string("forced solve failure")};
  };
}

double compute_l2_error(const std::vector<double> &numerical,
                        const std::vector<double> &analytical) {
  double error_sq = 0.0;
  for (std::size_t i = 0; i < numerical.size(); ++i) {
    const double diff = numerical[i] - analytical[i];
    error_sq += diff * diff;
  }
  return std::sqrt(error_sq / static_cast<double>(numerical.size()));
}

constexpr std::size_t LOCAL_SIZE = 64;

} // namespace

TEST_CASE("imex_euler_forward_euler_reduction", "[imex]") {
  const double c = 2.5;
  const double dt = 0.01;
  ConstantRHS rhs{c};
  auto solver = make_identity_solver();
  LinearOperatorDesc op_desc{"identity_noop"};

  std::vector<double> u_imex(LOCAL_SIZE, 1.0);
  std::vector<double> u_euler(LOCAL_SIZE, 1.0);

  ImexEulerStepper stepper(dt, LOCAL_SIZE, rhs, solver, op_desc);
  EulerStepper euler(dt, LOCAL_SIZE, rhs);

  MockExecutionService service;
  StageContext ctx{0.0, service};

  const auto attempt = stepper.attempt(0.0, u_imex, ctx);
  REQUIRE(attempt.success);
  REQUIRE(attempt.solve_status == ConvergenceStatus::converged);
  REQUIRE(stepper.commit(u_imex));
  (void)euler.step(0.0, u_euler);

  for (std::size_t i = 0; i < LOCAL_SIZE; ++i) {
    REQUIRE_THAT(u_imex[i], WithinAbs(u_euler[i], 1e-12));
  }
}

TEST_CASE("imex_euler_first_order_convergence", "[imex]") {
  // Manufactured ODE u' = -λu with split E=0 and L=-λ, so
  // (I - dt*L) = 1 + dt*λ and RHS = u_n → backward Euler u_{n+1}=u_n/(1+dtλ).
  const double lambda = 1.0;
  const double t_final = 0.1;
  const double u0 = 1.0;
  ZeroRHS E{};
  auto solver = make_diagonal_imex_solver();

  auto integrate = [&](double dt) {
    std::vector<double> diag(LOCAL_SIZE, 1.0 + dt * lambda);
    LinearOperatorDesc op_desc{"imex_diagonal", std::nullopt, diag};
    ImexEulerStepper stepper(dt, LOCAL_SIZE, E, solver, op_desc);
    std::vector<double> u(LOCAL_SIZE, u0);
    MockExecutionService service;
    StageContext ctx{0.0, service};
    double t = 0.0;
    const int steps = static_cast<int>(std::lround(t_final / dt));
    for (int i = 0; i < steps; ++i) {
      const auto attempt = stepper.attempt(t, u, ctx);
      REQUIRE(attempt.success);
      REQUIRE(stepper.commit(u));
      t = attempt.t_new;
    }
    return u;
  };

  const auto u_coarse = integrate(0.01);
  const auto u_fine = integrate(0.005);
  const auto u_finer = integrate(0.0025);

  const double exact = u0 * std::exp(-lambda * t_final);
  std::vector<double> analytical(LOCAL_SIZE, exact);

  const double err_coarse = compute_l2_error(u_coarse, analytical);
  const double err_fine = compute_l2_error(u_fine, analytical);
  const double err_finer = compute_l2_error(u_finer, analytical);

  REQUIRE(err_coarse > err_fine);
  REQUIRE(err_fine > err_finer);
  // First-order: halving dt roughly halves the error (allow some slack).
  REQUIRE(err_fine / err_coarse <= 0.6);
  REQUIRE(err_finer / err_fine <= 0.6);
}

TEST_CASE("imex_euler_failed_solve_preserves_accepted", "[imex]") {
  ConstantRHS rhs{1.0};
  auto solver = make_failing_solver();
  LinearOperatorDesc op_desc{"failing"};
  const double dt = 0.05;

  std::vector<double> u(LOCAL_SIZE, 3.14);
  const std::vector<double> u_before = u;

  ImexEulerStepper stepper(dt, LOCAL_SIZE, rhs, solver, op_desc);
  MockExecutionService service;
  StageContext ctx{0.0, service};

  const auto attempt = stepper.attempt(0.0, u, ctx);
  REQUIRE_FALSE(attempt.success);
  REQUIRE(attempt.solve_status == ConvergenceStatus::unknown_failure);
  REQUIRE(attempt.solve_failure_cause.has_value());
  REQUIRE(u == u_before);
  REQUIRE_FALSE(stepper.commit(u));
  REQUIRE(u == u_before);
}

TEST_CASE("imex_euler_multifield_bundle", "[imex]") {
  constexpr std::size_t N = 2;
  const double c1 = 2.0;
  const double c2 = 3.0;
  const double dt = 0.01;
  CompositeConstantRHS rhs{c1, c2};
  auto solver = make_identity_solver();
  LinearOperatorDesc op_desc{"identity_noop"};
  std::array<std::size_t, N> sizes{LOCAL_SIZE, LOCAL_SIZE};

  std::vector<double> u1(LOCAL_SIZE, 1.0);
  std::vector<double> u2(LOCAL_SIZE, 1.0);
  const std::vector<double> u1_initial = u1;
  const std::vector<double> u2_initial = u2;

  MultiImexEulerStepper<CompositeConstantRHS, decltype(solver), N> stepper(
      dt, sizes, rhs, solver, op_desc);

  MockExecutionService service;
  StageContext ctx{0.0, service};

  const auto attempt = stepper.attempt(0.0, ctx, u1, u2);
  REQUIRE(attempt.success);
  REQUIRE(stepper.commit(u1, u2));

  for (std::size_t i = 0; i < LOCAL_SIZE; ++i) {
    REQUIRE_THAT(u1[i], WithinAbs(u1_initial[i] + dt * c1, 1e-12));
    REQUIRE_THAT(u2[i], WithinAbs(u2_initial[i] + dt * c2, 1e-12));
  }
}
