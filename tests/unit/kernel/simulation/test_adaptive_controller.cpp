// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include <openpfc/kernel/integrator/error_evidence.hpp>
#include <openpfc/kernel/simulation/adaptive_controller.hpp>
#include <openpfc/kernel/simulation/steppers/butcher_tableau.hpp>
#include <openpfc/kernel/simulation/steppers/embedded_rk.hpp>
#include <openpfc/kernel/simulation/steppers/step_attempt.hpp>
#include <openpfc/kernel/simulation/time.hpp>

using pfc::Time;
using pfc::integrator::AggregationScope;
using pfc::integrator::make_embedded_pair_evidence;
using pfc::sim::AdaptiveControlConfig;
using pfc::sim::AdaptiveControlMode;
using pfc::sim::AdaptiveTimeController;
using pfc::sim::steppers::EmbeddedRKStepper;
using pfc::sim::steppers::commit_step_attempt;
using pfc::sim::steppers::make_embedded_rk23;

namespace {

AdaptiveControlConfig make_adaptive_cfg() {
  AdaptiveControlConfig cfg;
  cfg.mode = AdaptiveControlMode::adaptive;
  cfg.atol = 1e-5;
  cfg.rtol = 1e-5;
  cfg.safety_factor = 0.9;
  cfg.growth_max = 2.0;
  cfg.shrink_max = 0.5;
  cfg.min_dt = 1e-6;
  cfg.max_dt = 0.2;
  cfg.max_sequential_rejections = 20;
  return cfg;
}

struct PiecewiseStiffRhs {
  void operator()(double t, std::vector<double> &u,
                  std::vector<double> &du) const {
    const double k = (t < 0.15) ? 50.0 : 1.0;
    for (std::size_t i = 0; i < u.size(); ++i) {
      du[i] = -k * u[i];
    }
  }
};

} // namespace

TEST_CASE("fixed mode always accepts and keeps dt", "[adaptive_controller]") {
  AdaptiveControlConfig cfg;
  cfg.mode = AdaptiveControlMode::fixed;
  AdaptiveTimeController ctl(cfg, 3);
  const double norms[1] = {1e3};
  auto ev = make_embedded_pair_evidence(
      norms, AggregationScope::AlreadyReduced, 3);
  const auto d = ctl.decide(0.1, ev);
  REQUIRE(d.accepted);
  REQUIRE(d.next_dt == Catch::Approx(0.1));
}

TEST_CASE("large error rejects and shrinks dt", "[adaptive_controller]") {
  AdaptiveTimeController ctl(make_adaptive_cfg(), 3);
  const double norms[1] = {1.0}; // vs atol+rtol = 2e-5 → metric >> 1
  auto ev = make_embedded_pair_evidence(
      norms, AggregationScope::AlreadyReduced, 3);
  const auto d = ctl.decide(0.1, ev);
  REQUIRE_FALSE(d.accepted);
  REQUIRE(d.decision_available);
  REQUIRE(d.next_dt < 0.1);
  REQUIRE(d.next_dt >= make_adaptive_cfg().min_dt);
}

TEST_CASE("tiny error accepts and may grow dt", "[adaptive_controller]") {
  AdaptiveTimeController ctl(make_adaptive_cfg(), 3);
  const double norms[1] = {1e-12};
  auto ev = make_embedded_pair_evidence(
      norms, AggregationScope::AlreadyReduced, 3);
  const auto d = ctl.decide(0.05, ev);
  REQUIRE(d.accepted);
  REQUIRE(d.next_dt >= 0.05);
  REQUIRE(d.next_dt <= make_adaptive_cfg().max_dt);
}

TEST_CASE("apply commit/reject updates Time and counters",
          "[adaptive_controller]") {
  AdaptiveTimeController ctl(make_adaptive_cfg(), 3);
  Time time({0.0, 1.0, 0.1}, 0.0);

  time.begin_attempt(0.1);
  const double tiny[1] = {1e-12};
  auto ok = ctl.decide(time.get_attempted_dt(),
                       make_embedded_pair_evidence(
                           tiny, AggregationScope::AlreadyReduced, 3));
  REQUIRE(ok.accepted);
  ctl.apply(time, ok);
  REQUIRE(time.get_accepted_time() == Catch::Approx(0.1));
  REQUIRE(ctl.accepted_count() == 1);
  REQUIRE(time.get_accepted_steps() == 1);

  time.begin_attempt(time.get_dt());
  const double huge[1] = {1.0};
  auto bad = ctl.decide(time.get_attempted_dt(),
                        make_embedded_pair_evidence(
                            huge, AggregationScope::AlreadyReduced, 3));
  REQUIRE_FALSE(bad.accepted);
  const double t_before = time.get_accepted_time();
  ctl.apply(time, bad);
  REQUIRE(time.get_accepted_time() == Catch::Approx(t_before));
  REQUIRE(ctl.rejected_count() == 1);
  REQUIRE(time.get_rejected_steps() == 1);
}

TEST_CASE("embedded RK transient shrinks then grows",
          "[adaptive_controller][embedded_rk]") {
  PiecewiseStiffRhs rhs{};
  auto tableau = make_embedded_rk23<double>();
  EmbeddedRKStepper stepper(1, tableau, rhs);

  AdaptiveControlConfig cfg = make_adaptive_cfg();
  cfg.atol = 1e-4;
  cfg.rtol = 1e-4;
  cfg.max_dt = 0.08;
  AdaptiveTimeController ctl(cfg, /*error_order=*/3);

  Time time({0.0, 0.5, 0.05}, 0.0);
  std::vector<double> u{1.0};

  double min_dt_transient = 1.0;
  double max_dt_smooth = 0.0;
  int steps = 0;
  while (!time.done() && steps < 400) {
    time.begin_attempt(time.get_dt());
    const double dt = time.get_attempted_dt();
    const auto attempt = stepper.attempt(time.get_accepted_time(), dt, u);
    REQUIRE(attempt.success);
    const auto decision = ctl.decide_from_embedded_error(dt, stepper.error());
    if (decision.accepted) {
      commit_step_attempt(u, attempt);
    }
    ctl.apply(time, decision);
    const double now = time.get_accepted_time();
    if (now <= 0.15) {
      min_dt_transient = std::min(min_dt_transient, time.get_dt());
    } else {
      max_dt_smooth = std::max(max_dt_smooth, time.get_dt());
    }
    ++steps;
  }

  REQUIRE(time.done());
  REQUIRE(ctl.accepted_count() > 0);
  REQUIRE(ctl.rejected_count() > 0);
  REQUIRE(min_dt_transient < max_dt_smooth);
  REQUIRE(std::isfinite(u[0]));
}
