// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <tuple>
#include <type_traits>
#include <vector>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/kernel/simulation/solver_contract.hpp>
#include <openpfc/kernel/simulation/state_concepts.hpp>
#include <openpfc/kernel/simulation/steppers/butcher_tableau.hpp>
#include <openpfc/kernel/simulation/steppers/embedded_rk.hpp>
#include <openpfc/kernel/simulation/steppers/etd1.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>
#include <openpfc/kernel/simulation/steppers/explicit_rk.hpp>
#include <openpfc/kernel/simulation/steppers/imex_euler.hpp>
#include <openpfc/kernel/simulation/steppers/rk2_heun.hpp>
#include <openpfc/kernel/simulation/steppers/rk3_heun.hpp>
#include <openpfc/kernel/simulation/steppers/step_attempt.hpp>

using pfc::integrator::fill_spectral_exp_coeffs;
using pfc::sim::ConvergenceStatus;
using pfc::sim::LinearOperatorDesc;
using pfc::sim::SolveOptions;
using pfc::sim::SolveOutcome;
using pfc::sim::StageContext;
using pfc::sim::steppers::AttemptStepper;
using pfc::sim::steppers::commit_step_attempt;
using pfc::sim::steppers::EmbeddedRKStepper;
using pfc::sim::steppers::ETD1Stepper;
using pfc::sim::steppers::EulerStepper;
using pfc::sim::steppers::ExplicitRKStepper;
using pfc::sim::steppers::ImexEulerStepper;
using pfc::sim::steppers::make_embedded_rk23;
using pfc::sim::steppers::make_rk4_classical;
using pfc::sim::steppers::RK2HeunStepper;
using pfc::sim::steppers::RK3HeunStepper;
using pfc::sim::steppers::StepAttemptResult;

namespace {

struct DecayRhs {
  void operator()(double /*t*/, std::vector<double> &u,
                  std::vector<double> &du) const {
    for (std::size_t i = 0; i < u.size(); ++i) {
      du[i] = -u[i];
    }
  }
};

template <class Stepper>
void check_attempt_commit_rollback(Stepper &stepper, double t, double dt,
                                   std::vector<double> accepted) {
  const std::vector<double> before = accepted;
  const StepAttemptResult r = stepper.attempt(t, accepted);
  REQUIRE(accepted == before);
  REQUIRE(r.success);
  REQUIRE(r.t0 == Catch::Approx(t));
  REQUIRE(r.dt == Catch::Approx(dt));
  REQUIRE(r.t1 == Catch::Approx(t + dt));
  REQUIRE(r.candidate.size() == accepted.size());

  commit_step_attempt(accepted, r);
  REQUIRE(accepted == r.candidate);
  REQUIRE(accepted != before);

  const std::vector<double> after_commit = accepted;
  const StepAttemptResult r2 = stepper.attempt(r.t1, accepted);
  REQUIRE(accepted == after_commit);
  REQUIRE(r2.success);
  commit_step_attempt(accepted, r2);
  REQUIRE(accepted != after_commit);
}

} // namespace

TEST_CASE("EulerStepper attempt/commit leaves accepted state unchanged until commit",
          "[step_protocol][euler]") {
  DecayRhs rhs{};
  EulerStepper<DecayRhs> stepper(0.1, 4, rhs);
  static_assert(AttemptStepper<decltype(stepper)>);
  std::vector<double> u{1.0, 2.0, 3.0, 4.0};
  check_attempt_commit_rollback(stepper, 0.0, 0.1, u);
}

TEST_CASE(
    "RK2HeunStepper attempt/commit leaves accepted state unchanged until commit",
    "[step_protocol][rk2]") {
  DecayRhs rhs{};
  RK2HeunStepper<DecayRhs> stepper(0.1, 4, rhs);
  static_assert(AttemptStepper<decltype(stepper)>);
  std::vector<double> u{1.0, 2.0, 3.0, 4.0};
  check_attempt_commit_rollback(stepper, 0.0, 0.1, u);
}

TEST_CASE("EulerStepper::step matches attempt plus commit",
          "[step_protocol][euler]") {
  DecayRhs rhs{};
  EulerStepper<DecayRhs> a(0.25, 3, rhs);
  EulerStepper<DecayRhs> b(0.25, 3, rhs);
  std::vector<double> u_step{1.0, 0.5, -1.0};
  std::vector<double> u_attempt = u_step;
  const double t1 = a.step(1.0, u_step);
  const auto r = b.attempt(1.0, u_attempt);
  commit_step_attempt(u_attempt, r);
  REQUIRE(t1 == Catch::Approx(r.t1));
  REQUIRE(u_step == u_attempt);
}

TEST_CASE(
    "RK3HeunStepper attempt/commit leaves accepted state unchanged until commit",
    "[step_protocol][rk3]") {
  DecayRhs rhs{};
  RK3HeunStepper<DecayRhs> stepper(0.1, 4, rhs);
  static_assert(AttemptStepper<decltype(stepper)>);
  std::vector<double> u{1.0, 2.0, 3.0, 4.0};
  check_attempt_commit_rollback(stepper, 0.0, 0.1, u);
}

TEST_CASE(
    "ExplicitRKStepper attempt/commit leaves accepted state unchanged until commit",
    "[step_protocol][explicit_rk]") {
  DecayRhs rhs{};
  ExplicitRKStepper<DecayRhs> stepper(0.1, 4, make_rk4_classical<double>(), rhs);
  static_assert(AttemptStepper<decltype(stepper)>);
  std::vector<double> u{1.0, 2.0, 3.0, 4.0};
  check_attempt_commit_rollback(stepper, 0.0, 0.1, u);
}

TEST_CASE("ETD1Stepper attempt/commit leaves accepted state unchanged until commit",
          "[step_protocol][etd1]") {
  DecayRhs rhs{};
  ETD1Stepper<DecayRhs> stepper(0.1, 4, rhs);
  std::vector<double> L{-1.0, -1.0, -1.0, -1.0};
  std::vector<double> exp_buf(4);
  std::vector<double> phi_buf(4);
  fill_spectral_exp_coeffs(L, 0.1, exp_buf, phi_buf);
  stepper.set_coefficients(exp_buf, phi_buf);
  static_assert(AttemptStepper<decltype(stepper)>);
  std::vector<double> u{1.0, 2.0, 3.0, 4.0};
  check_attempt_commit_rollback(stepper, 0.0, 0.1, u);
}

TEST_CASE("EulerStepper attempt/commit on host Field<double>",
          "[step_protocol][euler][field]") {
  using pfc::data::Field;
  static_assert(pfc::field::Field<Field<double>>);
  static_assert(pfc::field::HostFieldState<Field<double>, double>);

  const auto domain = pfc::domain::create({2, 2, 1});
  const auto box = pfc::Box3i::from_bounds({0, 0, 0}, {1, 1, 0});
  Field<double> u(domain, box, 0);
  u.vec() = {1.0, 2.0, 3.0, 4.0};

  DecayRhs rhs{};
  EulerStepper<DecayRhs> stepper(0.1, u.size(), rhs);
  const auto before = u.vec();
  const StepAttemptResult r = stepper.attempt(0.0, u);
  REQUIRE(u.vec() == before);
  REQUIRE(r.success);
  commit_step_attempt(u.vec(), r);
  REQUIRE(u.vec() == r.candidate);
  REQUIRE(u.vec() != before);

  Field<double> v(domain, box, 0);
  v.vec() = before;
  (void)stepper.step(0.0, v);
  REQUIRE(v.vec() == u.vec());
}

struct ConstantComplexRhs {
  std::complex<double> c{};
  void operator()(double /*t*/, std::vector<std::complex<double>> & /*u*/,
                  std::vector<std::complex<double>> &du) const {
    for (auto &d : du) {
      d = c;
    }
  }
};

TEST_CASE("EulerStepper attempt/commit on host Field<complex>",
          "[step_protocol][euler][field][complex]") {
  using Complex = std::complex<double>;
  using pfc::data::Field;
  const auto domain = pfc::domain::create({2, 2, 1});
  const auto box = pfc::Box3i::from_bounds({0, 0, 0}, {1, 1, 0});
  Field<Complex> u(domain, box, 0);
  u.vec() = {Complex{1.0, 0.0}, Complex{0.0, 1.0}, Complex{-1.0, 0.5},
             Complex{0.25, -0.25}};
  ConstantComplexRhs rhs{Complex{0.1, -0.2}};
  EulerStepper<ConstantComplexRhs, Complex> stepper(0.5, u.size(), rhs);
  const auto before = u.vec();
  const auto r = stepper.attempt(0.0, u);
  REQUIRE(u.vec() == before);
  REQUIRE(r.success);
  commit_step_attempt(u.vec(), r);
  REQUIRE(u.vec() == r.candidate);
  REQUIRE(u.vec() != before);
}

TEST_CASE("RK2 and RK3 Heun complex constant RHS",
          "[step_protocol][rk2][rk3][complex]") {
  using Complex = std::complex<double>;
  constexpr Complex c{0.2, -0.1};
  constexpr Complex u0{1.0, 0.5};
  constexpr double dt = 0.25;
  ConstantComplexRhs rhs{c};
  const Complex expected = u0 + Complex(dt) * c;

  RK2HeunStepper<ConstantComplexRhs, Complex> rk2(dt, 1, rhs);
  std::vector<Complex> a{u0};
  const auto ra = rk2.attempt(0.0, a);
  REQUIRE(ra.success);
  REQUIRE(a[0].real() == Catch::Approx(u0.real()).margin(1e-14));
  REQUIRE(a[0].imag() == Catch::Approx(u0.imag()).margin(1e-14));
  REQUIRE(ra.candidate[0].real() == Catch::Approx(expected.real()).margin(1e-12));
  REQUIRE(ra.candidate[0].imag() == Catch::Approx(expected.imag()).margin(1e-12));

  RK3HeunStepper<ConstantComplexRhs, Complex> rk3(dt, 1, rhs);
  std::vector<Complex> b{u0};
  const auto rb = rk3.attempt(0.0, b);
  REQUIRE(rb.success);
  REQUIRE(rb.candidate[0].real() == Catch::Approx(expected.real()).margin(1e-12));
  REQUIRE(rb.candidate[0].imag() == Catch::Approx(expected.imag()).margin(1e-12));

  ExplicitRKStepper<ConstantComplexRhs, Complex> rk4(
      dt, 1, make_rk4_classical<double>(), rhs);
  std::vector<Complex> cvec{u0};
  const auto rc = rk4.attempt(0.0, cvec);
  REQUIRE(rc.success);
  REQUIRE(rc.candidate[0].real() == Catch::Approx(expected.real()).margin(1e-12));
  REQUIRE(rc.candidate[0].imag() == Catch::Approx(expected.imag()).margin(1e-12));
}

TEST_CASE("RK3HeunStepper attempt/commit on host Field<double>",
          "[step_protocol][rk3][field]") {
  using pfc::data::Field;
  const auto domain = pfc::domain::create({2, 2, 1});
  const auto box = pfc::Box3i::from_bounds({0, 0, 0}, {1, 1, 0});
  Field<double> u(domain, box, 0);
  u.vec() = {1.0, 2.0, 3.0, 4.0};
  DecayRhs rhs{};
  RK3HeunStepper<DecayRhs> stepper(0.1, u.size(), rhs);
  const auto before = u.vec();
  const StepAttemptResult r = stepper.attempt(0.0, u);
  REQUIRE(u.vec() == before);
  REQUIRE(r.success);
  commit_step_attempt(u.vec(), r);
  REQUIRE(u.vec() == r.candidate);
}

TEST_CASE("ExplicitRKStepper attempt/commit on host Field<double>",
          "[step_protocol][explicit_rk][field]") {
  using pfc::data::Field;
  const auto domain = pfc::domain::create({2, 2, 1});
  const auto box = pfc::Box3i::from_bounds({0, 0, 0}, {1, 1, 0});
  Field<double> u(domain, box, 0);
  u.vec() = {1.0, 2.0, 3.0, 4.0};
  DecayRhs rhs{};
  ExplicitRKStepper<DecayRhs> stepper(0.1, u.size(), make_rk4_classical<double>(),
                                      rhs);
  const auto before = u.vec();
  const StepAttemptResult r = stepper.attempt(0.0, u);
  REQUIRE(u.vec() == before);
  REQUIRE(r.success);
  commit_step_attempt(u.vec(), r);
  REQUIRE(u.vec() == r.candidate);
}

TEST_CASE(
    "EmbeddedRKStepper attempt/commit leaves accepted state unchanged until commit",
    "[step_protocol][embedded_rk]") {
  DecayRhs rhs{};
  EmbeddedRKStepper<DecayRhs> stepper(4, make_embedded_rk23<double>(), rhs);
  std::vector<double> u{1.0, 2.0, 3.0, 4.0};
  const std::vector<double> before = u;
  const StepAttemptResult r = stepper.attempt(0.0, 0.1, u);
  REQUIRE(u == before);
  REQUIRE(r.success);
  REQUIRE(r.t0 == Catch::Approx(0.0));
  REQUIRE(r.dt == Catch::Approx(0.1));
  REQUIRE(r.t1 == Catch::Approx(0.1));
  REQUIRE(r.candidate.size() == u.size());
  commit_step_attempt(u, r);
  REQUIRE(u == r.candidate);
  REQUIRE(u != before);

  const std::vector<double> after_commit = u;
  const StepAttemptResult r2 = stepper.attempt(r.t1, 0.1, u);
  REQUIRE(u == after_commit);
  REQUIRE(r2.success);
  commit_step_attempt(u, r2);
  REQUIRE(u != after_commit);
}

TEST_CASE(
    "ImexEulerStepper attempt/commit leaves accepted state unchanged until commit",
    "[step_protocol][imex]") {
  DecayRhs rhs{};
  auto solver = [](const LinearOperatorDesc &, const auto &rhs_bundle,
                   auto &target_bundle, const SolveOptions &, const StageContext &)
      -> SolveOutcome<std::decay_t<decltype(target_bundle)>> {
    using TargetType = std::decay_t<decltype(target_bundle)>;
    std::get<0>(target_bundle) = std::get<0>(rhs_bundle);
    return SolveOutcome<TargetType>{target_bundle, ConvergenceStatus::converged, 1,
                                    0.0, std::nullopt};
  };
  LinearOperatorDesc op_desc{"imex_identity", std::nullopt, std::monostate{}};
  ImexEulerStepper stepper(0.1, 4, rhs, solver, op_desc);
  StageContext ctx{};
  std::vector<double> u{1.0, 2.0, 3.0, 4.0};
  const std::vector<double> before = u;
  const StepAttemptResult r = stepper.attempt(0.0, u, ctx);
  REQUIRE(u == before);
  REQUIRE(r.success);
  REQUIRE(r.t0 == Catch::Approx(0.0));
  REQUIRE(r.dt == Catch::Approx(0.1));
  REQUIRE(r.t1 == Catch::Approx(0.1));
  REQUIRE(r.candidate.size() == u.size());
  REQUIRE(stepper.last_solve_status() == ConvergenceStatus::converged);
  commit_step_attempt(u, r);
  REQUIRE(u == r.candidate);
  REQUIRE(u != before);

  const std::vector<double> after_commit = u;
  const StepAttemptResult r2 = stepper.attempt(r.t1, u, ctx);
  REQUIRE(u == after_commit);
  REQUIRE(r2.success);
  commit_step_attempt(u, r2);
  REQUIRE(u != after_commit);
}

TEST_CASE("ETD1Stepper attempt on host Field<double>",
          "[step_protocol][etd1][field]") {
  using pfc::data::Field;
  const auto domain = pfc::domain::create({2, 2, 1});
  const auto box = pfc::Box3i::from_bounds({0, 0, 0}, {1, 1, 0});
  Field<double> u(domain, box, 0);
  u.vec() = {1.0, 2.0, 3.0, 4.0};
  DecayRhs rhs{};
  ETD1Stepper<DecayRhs> stepper(0.1, u.size(), rhs);
  std::vector<double> L(u.size(), -1.0);
  std::vector<double> exp_buf(u.size());
  std::vector<double> phi_buf(u.size());
  fill_spectral_exp_coeffs(L, 0.1, exp_buf, phi_buf);
  stepper.set_coefficients(exp_buf, phi_buf);
  const auto before = u.vec();
  const StepAttemptResult r = stepper.attempt(0.0, u);
  REQUIRE(u.vec() == before);
  REQUIRE(r.success);
  commit_step_attempt(u.vec(), r);
  REQUIRE(u.vec() == r.candidate);
}
