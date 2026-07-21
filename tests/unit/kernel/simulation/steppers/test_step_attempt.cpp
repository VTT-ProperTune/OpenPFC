// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/integrator/stage_context.hpp>
#include <openpfc/kernel/simulation/steppers/euler_attempt.hpp>
#include <openpfc/kernel/simulation/steppers/step_attempt.hpp>

using pfc::sim::steppers::commit_step_attempt;
using pfc::sim::steppers::EulerAttemptStepper;
using pfc::sim::steppers::MultiEulerAttemptStepper;

namespace {

struct ConstantRhs {
  double value{1.0};
  bool operator()(const pfc::integrator::StageContext & /*ctx*/,
                  const std::vector<double> &u,
                  std::vector<double> &du) const {
    for (std::size_t i = 0; i < u.size(); ++i) {
      du[i] = value;
    }
    return true;
  }
};

struct FailingEval {
  bool operator()(const pfc::integrator::StageContext & /*ctx*/,
                  const std::vector<double> &u,
                  std::vector<double> &du) const {
    // Soft-fail path must not require a filled du; leave unspecified.
    (void)u;
    (void)du;
    return false;
  }
};

struct OkPrep {
  bool operator()(const pfc::integrator::StageContext & /*ctx*/) const {
    return true;
  }
};

struct FailPrep {
  bool operator()(const pfc::integrator::StageContext & /*ctx*/) const {
    return false;
  }
};

struct IndependentIncrements2 {
  double du0{1.0};
  double du1{2.0};
  bool operator()(const pfc::integrator::StageContext & /*ctx*/,
                  const std::vector<double> &u0, const std::vector<double> &u1,
                  std::vector<double> &d0, std::vector<double> &d1) const {
    for (std::size_t i = 0; i < u0.size(); ++i) {
      d0[i] = du0;
    }
    for (std::size_t i = 0; i < u1.size(); ++i) {
      d1[i] = du1;
    }
    return true;
  }
};

} // namespace

TEST_CASE("success_isolates_accepted_until_commit", "[step_attempt][unit]") {
  ConstantRhs eval{1.0};
  OkPrep prep;
  EulerAttemptStepper<ConstantRhs, OkPrep> stepper(3);

  std::vector<double> accepted{1.0, 2.0, 3.0};
  const std::vector<double> fingerprint = accepted;
  const double t = 0.5;
  const double dt = 0.25;

  auto result = stepper.attempt(t, dt, accepted, eval, prep);

  REQUIRE(result.success);
  REQUIRE(result.t0 == Catch::Approx(t));
  REQUIRE(result.dt == Catch::Approx(dt));
  REQUIRE(result.t1 == Catch::Approx(t + dt));
  REQUIRE(accepted == fingerprint);
  REQUIRE(result.candidate.size() == accepted.size());
  for (std::size_t i = 0; i < accepted.size(); ++i) {
    REQUIRE(result.candidate[i] ==
            Catch::Approx(fingerprint[i] + dt * 1.0));
  }
  REQUIRE(stepper.workspace_reusable());

  commit_step_attempt(accepted, result);
  REQUIRE(accepted == result.candidate);
  REQUIRE(accepted != fingerprint);
}

TEST_CASE("failure_prep_leaves_accepted_unchanged", "[step_attempt][unit]") {
  ConstantRhs eval{1.0};
  FailPrep prep;
  EulerAttemptStepper<ConstantRhs, FailPrep> stepper(2);

  std::vector<double> accepted{4.0, -1.0};
  const std::vector<double> fingerprint = accepted;

  auto result = stepper.attempt(1.0, 0.1, accepted, eval, prep);

  REQUIRE_FALSE(result.success);
  REQUIRE(result.t1 == Catch::Approx(result.t0));
  REQUIRE(accepted == fingerprint);
  REQUIRE(stepper.workspace_reusable());
  REQUIRE_THROWS_AS(commit_step_attempt(accepted, result),
                    std::invalid_argument);
  REQUIRE(accepted == fingerprint);
}

TEST_CASE("failure_eval_leaves_accepted_unchanged", "[step_attempt][unit]") {
  FailingEval eval;
  OkPrep prep;
  EulerAttemptStepper<FailingEval, OkPrep> stepper(2);

  std::vector<double> accepted{0.5, 1.5};
  const std::vector<double> fingerprint = accepted;

  auto result = stepper.attempt(0.0, 0.2, accepted, eval, prep);

  REQUIRE_FALSE(result.success);
  REQUIRE(result.t1 == Catch::Approx(result.t0));
  REQUIRE(accepted == fingerprint);
  REQUIRE(stepper.workspace_reusable());
  REQUIRE_THROWS_AS(commit_step_attempt(accepted, result),
                    std::invalid_argument);
  REQUIRE(accepted == fingerprint);
}

TEST_CASE("multi_field_N2_isolation", "[step_attempt][unit]") {
  IndependentIncrements2 eval{1.0, 2.0};
  OkPrep prep;
  MultiEulerAttemptStepper<IndependentIncrements2, OkPrep> stepper(
      std::array<std::size_t, 2>{2, 3});

  std::vector<double> u0{1.0, 2.0};
  std::vector<double> u1{3.0, 4.0, 5.0};
  const std::vector<double> fp0 = u0;
  const std::vector<double> fp1 = u1;
  const double t = 0.0;
  const double dt = 0.5;

  auto result = stepper.attempt(t, dt, u0, u1, eval, prep);

  REQUIRE(result.success);
  REQUIRE(result.t1 == Catch::Approx(t + dt));
  REQUIRE(u0 == fp0);
  REQUIRE(u1 == fp1);
  REQUIRE(result.candidate(0).size() == 2);
  REQUIRE(result.candidate(1).size() == 3);
  for (std::size_t i = 0; i < fp0.size(); ++i) {
    REQUIRE(result.candidate(0)[i] == Catch::Approx(fp0[i] + dt * 1.0));
  }
  for (std::size_t i = 0; i < fp1.size(); ++i) {
    REQUIRE(result.candidate(1)[i] == Catch::Approx(fp1[i] + dt * 2.0));
  }
  REQUIRE(stepper.workspace_reusable());

  commit_step_attempt(u0, u1, result);
  REQUIRE(u0 == result.candidate(0));
  REQUIRE(u1 == result.candidate(1));
}
