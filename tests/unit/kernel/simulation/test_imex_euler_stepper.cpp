// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/simulation/steppers/euler.hpp>
#include <openpfc/kernel/simulation/steppers/imex_euler.hpp>

#include <array>
#include <cmath>
#include <complex>
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
  std::vector<double> global_reduce(const std::vector<double> &data,
                                    MPI_Op) override {
    // Return copy of input (serial behavior)
    return data;
  }
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
  void
  operator()(double /*t*/,
             std::tuple<std::vector<double> &, std::vector<double> &> /*u*/,
             std::tuple<std::vector<double> &, std::vector<double> &> du) const {
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
    return SolveOutcome<TargetType>{target, ConvergenceStatus::converged, 1, 0.0,
                                    std::nullopt};
  };
}

// Dense n×n (row-major) linear solve from operator_context. Proves
// SolveFunction / ImexEuler do not assume a diagonal implicit operator.
auto make_dense_nondiagonal_solver() {
  return [](const LinearOperatorDesc &op_desc, const auto &rhs_bundle,
            auto &target_bundle, const SolveOptions &, const StageContext &)
             -> SolveOutcome<std::decay_t<decltype(target_bundle)>> {
    using TargetType = std::decay_t<decltype(target_bundle)>;
    if (!std::holds_alternative<std::vector<double>>(op_desc.operator_context)) {
      return SolveOutcome<TargetType>{
          target_bundle, ConvergenceStatus::ill_conditioned, 0, 0.0,
          std::string("dense solver requires vector operator_context")};
    }
    const auto &Aflat = std::get<std::vector<double>>(op_desc.operator_context);
    const auto &rhs_vec = std::get<0>(rhs_bundle);
    auto &x = std::get<0>(target_bundle);
    const std::size_t n = rhs_vec.size();
    if (n == 0 || Aflat.size() != n * n || x.size() != n) {
      return SolveOutcome<TargetType>{target_bundle,
                                      ConvergenceStatus::ill_conditioned, 0, 0.0,
                                      std::string("dense solver size mismatch")};
    }
    std::vector<std::vector<double>> A(n, std::vector<double>(n + 1, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        A[i][j] = Aflat[i * n + j];
      }
      A[i][n] = rhs_vec[i];
    }
    for (std::size_t k = 0; k < n; ++k) {
      std::size_t piv = k;
      for (std::size_t i = k + 1; i < n; ++i) {
        if (std::abs(A[i][k]) > std::abs(A[piv][k])) {
          piv = i;
        }
      }
      if (std::abs(A[piv][k]) < 1e-14) {
        return SolveOutcome<TargetType>{target_bundle,
                                        ConvergenceStatus::ill_conditioned, 0, 0.0,
                                        std::string("dense solver singular pivot")};
      }
      std::swap(A[k], A[piv]);
      const double diag = A[k][k];
      for (std::size_t j = k; j <= n; ++j) {
        A[k][j] /= diag;
      }
      for (std::size_t i = 0; i < n; ++i) {
        if (i == k) {
          continue;
        }
        const double f = A[i][k];
        for (std::size_t j = k; j <= n; ++j) {
          A[i][j] -= f * A[k][j];
        }
      }
    }
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = A[i][n];
    }
    return SolveOutcome<TargetType>{target_bundle, ConvergenceStatus::converged,
                                    static_cast<int>(n), 0.0, std::nullopt};
  };
}

auto make_failing_solver() {
  return [](const LinearOperatorDesc &, const auto & /*rhs*/, auto &target,
            const SolveOptions &,
            const StageContext &) -> SolveOutcome<std::decay_t<decltype(target)>> {
    using TargetType = std::decay_t<decltype(target)>;
    return SolveOutcome<TargetType>{target, ConvergenceStatus::unknown_failure, 0,
                                    1.0, std::string("forced solve failure")};
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
  StageContext ctx{.time = 0.0, .execution_service = &service};

  const auto attempt = stepper.attempt(0.0, u_imex, ctx);
  REQUIRE(attempt.success);
  REQUIRE(stepper.last_solve_status() == ConvergenceStatus::converged);
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
    StageContext ctx{.time = 0.0, .execution_service = &service};
    double t = 0.0;
    const int steps = static_cast<int>(std::lround(t_final / dt));
    for (int i = 0; i < steps; ++i) {
      const auto attempt = stepper.attempt(t, u, ctx);
      REQUIRE(attempt.success);
      REQUIRE(stepper.commit(u));
      t = attempt.t1;
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
  StageContext ctx{.time = 0.0, .execution_service = &service};

  const auto attempt = stepper.attempt(0.0, u, ctx);
  REQUIRE_FALSE(attempt.success);
  REQUIRE(stepper.last_solve_status() == ConvergenceStatus::unknown_failure);
  REQUIRE(stepper.last_solve_failure_cause().has_value());
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
  StageContext ctx{.time = 0.0, .execution_service = &service};

  const auto attempt = stepper.attempt(0.0, ctx, u1, u2);
  REQUIRE(attempt.success);
  REQUIRE(stepper.commit(u1, u2));

  for (std::size_t i = 0; i < LOCAL_SIZE; ++i) {
    REQUIRE_THAT(u1[i], WithinAbs(u1_initial[i] + dt * c1, 1e-12));
    REQUIRE_THAT(u2[i], WithinAbs(u2_initial[i] + dt * c2, 1e-12));
  }
}

TEST_CASE("imex_euler_nondiagonal_dense_solve", "[imex]") {
  // Implicit half is the coupled 2×2 operator L = [[0, 1], [1, 0]].
  // With E = 0, IMEX Euler solves (I - dt L) u_{n+1} = u_n.
  constexpr double dt = 0.1;
  ZeroRHS E{};
  auto solver = make_dense_nondiagonal_solver();
  const std::vector<double> A{1.0, -dt, -dt, 1.0};
  LinearOperatorDesc op_desc{"imex_dense", std::nullopt, A};

  std::vector<double> u{1.0, 0.0};
  const std::vector<double> u_before = u;

  ImexEulerStepper stepper(dt, 2, E, solver, op_desc);
  MockExecutionService service;
  StageContext ctx{.time = 0.0, .execution_service = &service};

  const auto attempt = stepper.attempt(0.0, u, ctx);
  REQUIRE(attempt.success);
  REQUIRE(stepper.last_solve_status() == ConvergenceStatus::converged);
  REQUIRE(u == u_before);
  REQUIRE(stepper.commit(u));

  const double den = 1.0 - dt * dt;
  REQUIRE_THAT(u[0], WithinAbs(1.0 / den, 1e-12));
  REQUIRE_THAT(u[1], WithinAbs(dt / den, 1e-12));
}

using Complex = std::complex<double>;

struct ConstantComplexRHS {
  Complex c{};
  void operator()(double /*t*/, std::vector<Complex> & /*u*/,
                  std::vector<Complex> &du) const {
    for (auto &v : du) {
      v = c;
    }
  }
};

struct TwoFieldConstantComplexRHS {
  Complex c0{};
  Complex c1{};
  void
  operator()(double /*t*/,
             std::tuple<std::vector<Complex> &, std::vector<Complex> &> /*u*/,
             std::tuple<std::vector<Complex> &, std::vector<Complex> &> du) const {
    std::get<0>(du)[0] = c0;
    std::get<1>(du)[0] = c1;
  }
};

TEST_CASE("imex_euler_complex_identity_matches_euler", "[imex][complex]") {
  constexpr Complex c{0.25, -0.5};
  constexpr Complex u0{1.0, 2.0};
  constexpr double dt = 0.1;
  ConstantComplexRHS rhs{c};
  auto solver = make_identity_solver();
  LinearOperatorDesc op_desc{"identity_noop"};

  ImexEulerStepper<ConstantComplexRHS, decltype(solver), Complex> imex(
      dt, 1, rhs, solver, op_desc);
  EulerStepper<ConstantComplexRHS, Complex> euler(dt, 1, rhs);

  MockExecutionService service;
  StageContext ctx{.time = 0.0, .execution_service = &service};

  std::vector<Complex> u_imex{u0};
  std::vector<Complex> u_euler{u0};
  const auto fingerprint = u_imex;
  const auto attempt = imex.attempt(0.0, u_imex, ctx);
  REQUIRE(attempt.success);
  REQUIRE(u_imex == fingerprint);
  REQUIRE(imex.commit(u_imex));
  (void)euler.step(0.0, u_euler);

  const Complex expected = u0 + Complex(dt) * c;
  REQUIRE(u_imex[0].real() == Catch::Approx(expected.real()).margin(1e-12));
  REQUIRE(u_imex[0].imag() == Catch::Approx(expected.imag()).margin(1e-12));
  REQUIRE(u_euler[0].real() == Catch::Approx(expected.real()).margin(1e-12));
  REQUIRE(u_euler[0].imag() == Catch::Approx(expected.imag()).margin(1e-12));
}

TEST_CASE("imex_euler_complex_diagonal_implicit", "[imex][complex]") {
  // E = 0, L = -λ (real): (I - dt L) = 1 + dt λ, u_{n+1} = u_n / (1+dtλ).
  constexpr double lambda = 2.0;
  constexpr double dt = 0.05;
  constexpr Complex u0{0.5, -0.25};
  struct ZeroComplexRHS {
    void operator()(double /*t*/, std::vector<Complex> & /*u*/,
                    std::vector<Complex> &du) const {
      for (auto &v : du) {
        v = Complex{};
      }
    }
  };
  ZeroComplexRHS E{};
  auto solver = make_diagonal_imex_solver();
  std::vector<double> diag{1.0 + dt * lambda};
  LinearOperatorDesc op_desc{"imex_diagonal", std::nullopt, diag};
  ImexEulerStepper<ZeroComplexRHS, decltype(solver), Complex> stepper(
      dt, 1, E, solver, op_desc);
  MockExecutionService service;
  StageContext ctx{.time = 0.0, .execution_service = &service};
  std::vector<Complex> u{u0};
  const auto attempt = stepper.attempt(0.0, u, ctx);
  REQUIRE(attempt.success);
  REQUIRE(stepper.commit(u));
  const Complex expected = u0 / (1.0 + dt * lambda);
  REQUIRE(u[0].real() == Catch::Approx(expected.real()).margin(1e-12));
  REQUIRE(u[0].imag() == Catch::Approx(expected.imag()).margin(1e-12));
}

TEST_CASE("imex_euler_complex_multifield", "[imex][complex]") {
  constexpr double dt = 0.2;
  constexpr Complex c0{0.1, 0.2};
  constexpr Complex c1{-0.3, 0.05};
  TwoFieldConstantComplexRHS rhs{c0, c1};
  auto solver = make_identity_solver();
  LinearOperatorDesc op_desc{"identity_noop"};
  MultiImexEulerStepper<TwoFieldConstantComplexRHS, decltype(solver), 2, Complex>
      stepper(dt, {1, 1}, rhs, solver, op_desc);
  MockExecutionService service;
  StageContext ctx{.time = 0.0, .execution_service = &service};
  std::vector<Complex> a{Complex{1.0, 0.0}};
  std::vector<Complex> b{Complex{0.0, 1.0}};
  const auto fa = a;
  const auto fb = b;
  const auto attempt = stepper.attempt(0.0, ctx, a, b);
  REQUIRE(attempt.success);
  REQUIRE(a == fa);
  REQUIRE(b == fb);
  REQUIRE(stepper.commit(a, b));
  const Complex e0 = fa[0] + Complex(dt) * c0;
  const Complex e1 = fb[0] + Complex(dt) * c1;
  REQUIRE(a[0].real() == Catch::Approx(e0.real()).margin(1e-12));
  REQUIRE(b[0].imag() == Catch::Approx(e1.imag()).margin(1e-12));
}
