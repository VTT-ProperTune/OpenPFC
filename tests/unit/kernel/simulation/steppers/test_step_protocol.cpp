// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>
#include <vector>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/kernel/simulation/state_concepts.hpp>
#include <openpfc/kernel/simulation/steppers/butcher_tableau.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>
#include <openpfc/kernel/simulation/steppers/etd1.hpp>
#include <openpfc/kernel/simulation/steppers/explicit_rk.hpp>
#include <openpfc/kernel/simulation/steppers/rk2_heun.hpp>
#include <openpfc/kernel/simulation/steppers/rk3_heun.hpp>
#include <openpfc/kernel/simulation/steppers/step_attempt.hpp>

using pfc::integrator::fill_spectral_exp_coeffs;
using pfc::sim::steppers::AttemptStepper;
using pfc::sim::steppers::commit_step_attempt;
using pfc::sim::steppers::Etd1Stepper;
using pfc::sim::steppers::EulerStepper;
using pfc::sim::steppers::ExplicitRKStepper;
using pfc::sim::steppers::RK2HeunStepper;
using pfc::sim::steppers::RK3HeunStepper;
using pfc::sim::steppers::StepAttemptResult;
using pfc::sim::steppers::make_rk4_classical;

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

TEST_CASE("RK2HeunStepper attempt/commit leaves accepted state unchanged until commit",
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

TEST_CASE("RK3HeunStepper attempt/commit leaves accepted state unchanged until commit",
          "[step_protocol][rk3]") {
  DecayRhs rhs{};
  RK3HeunStepper<DecayRhs> stepper(0.1, 4, rhs);
  static_assert(AttemptStepper<decltype(stepper)>);
  std::vector<double> u{1.0, 2.0, 3.0, 4.0};
  check_attempt_commit_rollback(stepper, 0.0, 0.1, u);
}

TEST_CASE("ExplicitRKStepper attempt/commit leaves accepted state unchanged until commit",
          "[step_protocol][explicit_rk]") {
  DecayRhs rhs{};
  ExplicitRKStepper<DecayRhs> stepper(0.1, 4, make_rk4_classical<double>(), rhs);
  static_assert(AttemptStepper<decltype(stepper)>);
  std::vector<double> u{1.0, 2.0, 3.0, 4.0};
  check_attempt_commit_rollback(stepper, 0.0, 0.1, u);
}

TEST_CASE("Etd1Stepper attempt/commit leaves accepted state unchanged until commit",
          "[step_protocol][etd1]") {
  DecayRhs rhs{};
  Etd1Stepper<DecayRhs> stepper(0.1, 4, rhs);
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

TEST_CASE("Etd1Stepper attempt on host Field<double>",
          "[step_protocol][etd1][field]") {
  using pfc::data::Field;
  const auto domain = pfc::domain::create({2, 2, 1});
  const auto box = pfc::Box3i::from_bounds({0, 0, 0}, {1, 1, 0});
  Field<double> u(domain, box, 0);
  u.vec() = {1.0, 2.0, 3.0, 4.0};
  DecayRhs rhs{};
  Etd1Stepper<DecayRhs> stepper(0.1, u.size(), rhs);
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
