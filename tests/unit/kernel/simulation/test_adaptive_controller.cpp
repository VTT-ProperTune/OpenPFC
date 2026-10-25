// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <optional>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/integrator/error_evidence.hpp>
#include <openpfc/kernel/simulation/adaptive_controller.hpp>
#include <openpfc/kernel/simulation/steppers/butcher_tableau.hpp>
#include <openpfc/kernel/simulation/steppers/embedded_rk.hpp>
#include <openpfc/kernel/simulation/steppers/step_attempt.hpp>
#include <openpfc/kernel/simulation/time.hpp>

using pfc::Time;
using pfc::integrator::AggregationScope;
using pfc::integrator::ErrorEvidence;
using pfc::integrator::make_embedded_pair_evidence;
using pfc::integrator::make_invalid_evidence;
using pfc::integrator::SolutionScale;
using pfc::sim::AdaptiveControlConfig;
using pfc::sim::AdaptiveControllerState;
using pfc::sim::AdaptiveControlMode;
using pfc::sim::AdaptiveTimeController;
using pfc::sim::StepController;
using pfc::sim::steppers::commit_step_attempt;
using pfc::sim::steppers::EmbeddedRKStepper;
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
  void operator()(double t, std::vector<double> &u, std::vector<double> &du) const {
    const double k = (t < 0.15) ? 50.0 : 1.0;
    for (std::size_t i = 0; i < u.size(); ++i) {
      du[i] = -k * u[i];
    }
  }
};

/// Unit scale keeps the historical `atol + rtol * 1` denominator explicit.
ErrorEvidence unit_norm(double norm) {
  const double norms[1] = {norm};
  return make_embedded_pair_evidence(norms, AggregationScope::AlreadyReduced, 3,
                                     std::nullopt, SolutionScale::Unit);
}

ErrorEvidence scaled_norm(double norm, double scale) {
  const double norms[1] = {norm};
  const double weights[1] = {scale};
  return make_embedded_pair_evidence(norms, AggregationScope::AlreadyReduced, 3,
                                     std::span<const double>{weights});
}

void accept_metric(AdaptiveTimeController &ctl, Time &time, double metric) {
  const double den = 1e-5 + 1e-5;
  time.begin_attempt(time.get_dt());
  const auto d = ctl.decide(time.get_attempted_dt(), unit_norm(metric * den));
  REQUIRE(d.accepted);
  REQUIRE(d.metric == Catch::Approx(metric));
  ctl.apply(time, d);
}

} // namespace

TEST_CASE("fixed mode always accepts and keeps dt", "[adaptive_controller]") {
  AdaptiveControlConfig cfg;
  cfg.mode = AdaptiveControlMode::fixed;
  AdaptiveTimeController ctl(cfg, 3);
  const double norms[1] = {1e3};
  auto ev = make_embedded_pair_evidence(norms, AggregationScope::AlreadyReduced, 3);
  const auto d = ctl.decide(0.1, ev);
  REQUIRE(d.accepted);
  REQUIRE(d.next_dt == Catch::Approx(0.1));
}

TEST_CASE("large error rejects and shrinks dt", "[adaptive_controller]") {
  AdaptiveTimeController ctl(make_adaptive_cfg(), 3);
  // Unit scale: atol + rtol * 1 = 2e-5, so a norm of 1 is metric >> 1.
  const auto d = ctl.decide(0.1, unit_norm(1.0));
  REQUIRE_FALSE(d.accepted);
  REQUIRE(d.decision_available);
  REQUIRE(d.next_dt < 0.1);
  REQUIRE(d.next_dt >= make_adaptive_cfg().min_dt);
}

TEST_CASE("tiny error accepts and may grow dt", "[adaptive_controller]") {
  AdaptiveTimeController ctl(make_adaptive_cfg(), 3);
  const auto d = ctl.decide(0.05, unit_norm(1e-12));
  REQUIRE(d.accepted);
  REQUIRE(d.next_dt >= 0.05);
  REQUIRE(d.next_dt <= make_adaptive_cfg().max_dt);
}

TEST_CASE("apply commit/reject updates Time and counters", "[adaptive_controller]") {
  AdaptiveTimeController ctl(make_adaptive_cfg(), 3);
  Time time({0.0, 1.0, 0.1}, 0.0);

  time.begin_attempt(0.1);
  auto ok = ctl.decide(time.get_attempted_dt(), unit_norm(1e-12));
  REQUIRE(ok.accepted);
  ctl.apply(time, ok);
  REQUIRE(time.get_accepted_time() == Catch::Approx(0.1));
  REQUIRE(ctl.accepted_count() == 1);
  REQUIRE(time.get_accepted_steps() == 1);

  time.begin_attempt(time.get_dt());
  auto bad = ctl.decide(time.get_attempted_dt(), unit_norm(1.0));
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
    const auto decision =
        ctl.decide_from_embedded_error(dt, stepper.error(), u, stepper.u_high());
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

TEST_CASE("relative tolerance without a scale fails closed",
          "[adaptive_controller]") {
  AdaptiveTimeController ctl(make_adaptive_cfg(), 3);
  const double norms[1] = {1e-12};
  const auto ev =
      make_embedded_pair_evidence(norms, AggregationScope::AlreadyReduced, 3);
  Time time({0.0, 1.0, 0.1}, 0.0);
  time.begin_attempt(0.1);
  const auto d = ctl.decide(0.1, ev);
  REQUIRE_FALSE(d.accepted);
  REQUIRE_FALSE(d.decision_available);
  ctl.apply(time, d);
  REQUIRE(time.get_accepted_time() == Catch::Approx(0.0));
  REQUIRE_FALSE(ctl.has_previous_accepted_metric());
}

TEST_CASE("pure atol ignores the solution scale", "[adaptive_controller]") {
  AdaptiveControlConfig cfg = make_adaptive_cfg();
  cfg.atol = 1e-4;
  cfg.rtol = 0.0;
  AdaptiveTimeController ctl(cfg, 3);
  const auto d = ctl.decide(0.1, scaled_norm(1e-5, /*scale=*/1e6));
  REQUIRE(d.accepted);
  REQUIRE(d.metric == Catch::Approx(0.1));
}

TEST_CASE("solution scale changes a pure relative metric", "[adaptive_controller]") {
  AdaptiveControlConfig cfg = make_adaptive_cfg();
  cfg.atol = 0.0;
  cfg.rtol = 1e-2;
  cfg.growth_max = 4.0;
  cfg.shrink_max = 0.2;
  AdaptiveTimeController ctl(cfg, 1);
  // den = 1e-2 * 2, norm 1e-2 → metric 0.5.
  const auto accept = ctl.decide(0.1, scaled_norm(1e-2, 2.0));
  REQUIRE(accept.accepted);
  REQUIRE(accept.metric == Catch::Approx(0.5));
  // Same norm against scale 0.25 → metric 4.
  const auto reject = ctl.decide(0.1, scaled_norm(1e-2, 0.25));
  REQUIRE_FALSE(reject.accepted);
  REQUIRE(reject.metric == Catch::Approx(4.0));
}

TEST_CASE("mixed atol and rtol use the supplied scale", "[adaptive_controller]") {
  AdaptiveControlConfig cfg = make_adaptive_cfg();
  cfg.atol = 1e-4;
  cfg.rtol = 1e-2;
  AdaptiveTimeController ctl(cfg, 3);
  // den = 1e-4 + 1e-2 * 1 = 0.0101.
  const auto on_unit = ctl.decide(0.1, scaled_norm(0.0101, 1.0));
  REQUIRE(on_unit.metric == Catch::Approx(1.0));
  // den = 1e-4 + 1e-2 * 10 = 0.1001.
  const auto on_large = ctl.decide(0.1, scaled_norm(0.0101, 10.0));
  REQUIRE(on_large.accepted);
  REQUIRE(on_large.metric == Catch::Approx(0.0101 / 0.1001));
}

TEST_CASE("several fields take the max scaled error", "[adaptive_controller]") {
  AdaptiveControlConfig cfg = make_adaptive_cfg();
  cfg.atol = 0.0;
  cfg.rtol = 1e-4;
  cfg.field_count = 2;
  AdaptiveTimeController ctl(cfg, 3);
  const double norms[2] = {1e-4, 2e-3};
  const double scales[2] = {1.0, 10.0};
  const auto ev = make_embedded_pair_evidence(
      norms, AggregationScope::AlreadyReduced, 3, std::span<const double>{scales});
  const auto d = ctl.decide(0.1, ev);
  // e0 = 1, e1 = 2.
  REQUIRE_FALSE(d.accepted);
  REQUIRE(d.metric == Catch::Approx(2.0));
}

TEST_CASE("per-field tolerances are applied field by field",
          "[adaptive_controller]") {
  AdaptiveControlConfig cfg = make_adaptive_cfg();
  cfg.atol = 0.0;
  cfg.rtol = 0.0;
  cfg.atol_per_field = {1.0, 1e-3};
  cfg.rtol_per_field = {0.0, 0.0};
  cfg.field_count = 2;
  AdaptiveTimeController ctl(cfg, 3);
  const double norms[2] = {0.1, 0.1};
  const double scales[2] = {1.0, 1.0};
  const auto ev = make_embedded_pair_evidence(
      norms, AggregationScope::AlreadyReduced, 3, std::span<const double>{scales});
  const auto d = ctl.decide(0.1, ev);
  REQUIRE_FALSE(d.accepted);
  REQUIRE(d.metric == Catch::Approx(100.0));
}

TEST_CASE("embedded error is scaled component-wise", "[adaptive_controller]") {
  AdaptiveControlConfig cfg = make_adaptive_cfg();
  cfg.atol = 0.0;
  cfg.rtol = 1.0;
  cfg.shrink_max = 0.1;
  cfg.growth_max = 4.0;
  AdaptiveTimeController ctl(cfg, 1);
  const double error[2] = {1e-8, 1.0};
  const double accepted[2] = {1.0, 1e-8};
  const double candidate[2] = {1.0, 1e-8};
  const auto d = ctl.decide_from_embedded_error(0.2, error, accepted, candidate);
  // Component-wise max is 1/1e-8. An inf-norm of the whole vector would be 1.
  REQUIRE_FALSE(d.accepted);
  REQUIRE(d.metric == Catch::Approx(1e8));
  REQUIRE(d.next_dt == Catch::Approx(0.02)); // shrink_max 0.1 * 0.2
}

TEST_CASE("embedded zero state uses atol and rejects pure rtol",
          "[adaptive_controller]") {
  AdaptiveControlConfig absolute = make_adaptive_cfg();
  absolute.atol = 1e-6;
  absolute.rtol = 1e-4;
  AdaptiveTimeController abs_ctl(absolute, 3);
  const double error[1] = {1e-8};
  const double zero[1] = {0.0};
  const auto with_atol = abs_ctl.decide_from_embedded_error(0.1, error, zero, zero);
  REQUIRE(with_atol.accepted);
  REQUIRE(with_atol.metric == Catch::Approx(0.01));

  AdaptiveControlConfig relative = make_adaptive_cfg();
  relative.atol = 0.0;
  relative.rtol = 1e-4;
  AdaptiveTimeController rel_ctl(relative, 3);
  const auto pure = rel_ctl.decide_from_embedded_error(0.1, error, zero, zero);
  REQUIRE_FALSE(pure.accepted);
  REQUIRE(pure.decision_available);
  REQUIRE(std::isinf(pure.metric));
}

TEST_CASE("embedded near-zero state makes a large relative metric",
          "[adaptive_controller]") {
  AdaptiveControlConfig cfg = make_adaptive_cfg();
  cfg.atol = 0.0;
  cfg.rtol = 1e-6;
  AdaptiveTimeController ctl(cfg, 3);
  const double error[1] = {1e-8};
  const double tiny[1] = {1e-16};
  const auto d = ctl.decide_from_embedded_error(0.1, error, tiny, tiny);
  REQUIRE_FALSE(d.accepted);
  REQUIRE(d.metric == Catch::Approx(1e14));
}

TEST_CASE("memoryless exponent is one over error_order", "[adaptive_controller]") {
  AdaptiveControlConfig cfg = make_adaptive_cfg();
  cfg.safety_factor = 1.0;
  cfg.growth_max = 100.0;
  cfg.shrink_max = 0.01;
  cfg.min_dt = 1e-12;
  cfg.max_dt = 100.0;
  AdaptiveTimeController ctl(cfg, /*error_order=*/1);
  const auto d = ctl.decide(1.0, unit_norm(0.25 * 2e-5));
  REQUIRE(d.accepted);
  // metric 0.25, factor = 1/0.25 = 4. Order+1 would give 2.
  REQUIRE(d.next_dt == Catch::Approx(4.0));
}

TEST_CASE("growth shrink and dt bounds clip the proposal", "[adaptive_controller]") {
  AdaptiveControlConfig cfg = make_adaptive_cfg();
  cfg.safety_factor = 1.0;
  cfg.growth_max = 2.0;
  cfg.shrink_max = 0.5;
  cfg.min_dt = 1e-12;
  cfg.max_dt = 100.0;
  AdaptiveTimeController ctl(cfg, 1);

  const auto grown = ctl.decide(1.0, unit_norm(0.01 * 2e-5));
  REQUIRE(grown.accepted);
  REQUIRE(grown.next_dt == Catch::Approx(2.0)); // raw factor 100, clip growth_max

  const auto shrunk = ctl.decide(1.0, unit_norm(100.0 * 2e-5));
  REQUIRE_FALSE(shrunk.accepted);
  REQUIRE(shrunk.next_dt == Catch::Approx(0.5)); // raw factor 0.01, clip shrink_max

  cfg.min_dt = 0.01;
  cfg.max_dt = 3.0;
  cfg.growth_max = 10.0;
  cfg.shrink_max = 0.1;
  AdaptiveTimeController bounded(cfg, 1);
  const auto at_max = bounded.decide(1.0, unit_norm(1e-6 * 2e-5));
  REQUIRE(at_max.next_dt == Catch::Approx(3.0));
  const auto at_min = bounded.decide(0.02, unit_norm(100.0 * 2e-5));
  REQUIRE(at_min.next_dt == Catch::Approx(0.01));
}

TEST_CASE("PI history follows accepted metrics only", "[adaptive_controller]") {
  AdaptiveControlConfig cfg = make_adaptive_cfg();
  cfg.controller = StepController::pi;
  cfg.safety_factor = 0.9;
  cfg.growth_max = 5.0;
  cfg.shrink_max = 0.1;
  cfg.min_dt = 1e-8;
  cfg.max_dt = 10.0;
  AdaptiveTimeController pi(cfg, 3);
  AdaptiveControlConfig memoryless_cfg = cfg;
  memoryless_cfg.controller = StepController::memoryless;
  AdaptiveTimeController memoryless(memoryless_cfg, 3);

  Time time({0.0, 10.0, 0.1}, 0.0);
  time.begin_attempt(0.1);
  const auto first = pi.decide(0.1, unit_norm(0.25 * 2e-5));
  const auto first_memoryless = memoryless.decide(0.1, unit_norm(0.25 * 2e-5));
  REQUIRE(first.accepted);
  REQUIRE(first.next_dt == Catch::Approx(first_memoryless.next_dt));
  REQUIRE(first.next_dt == Catch::Approx(0.1 * 0.9 * std::pow(0.25, -1.0 / 3.0)));
  pi.apply(time, first);
  REQUIRE(pi.has_previous_accepted_metric());
  REQUIRE(pi.previous_accepted_metric() == Catch::Approx(0.25));

  const double pi_factor =
      0.9 * std::pow(0.25, -0.7 / 3.0) * std::pow(0.25, 0.4 / 3.0);
  time.begin_attempt(0.1);
  const auto second = pi.decide(0.1, unit_norm(0.25 * 2e-5));
  REQUIRE(second.accepted);
  REQUIRE(second.next_dt == Catch::Approx(0.1 * pi_factor));
  REQUIRE(second.next_dt != Catch::Approx(first.next_dt));
  pi.apply(time, second);

  const double previous = pi.previous_accepted_metric();
  time.begin_attempt(0.1);
  const auto rejected = pi.decide(0.1, unit_norm(4.0 * 2e-5));
  REQUIRE_FALSE(rejected.accepted);
  const double t_before = time.get_accepted_time();
  pi.apply(time, rejected);
  REQUIRE(time.get_accepted_time() == Catch::Approx(t_before));
  REQUIRE(pi.previous_accepted_metric() == Catch::Approx(previous));
  REQUIRE(rejected.next_dt == Catch::Approx(0.1 * 0.9 * std::pow(4.0, -1.0 / 3.0)));

  time.begin_attempt(0.1);
  const auto after_reject = pi.decide(0.1, unit_norm(0.25 * 2e-5));
  REQUIRE(after_reject.next_dt == Catch::Approx(second.next_dt));
}

TEST_CASE("PI gains of one and zero match the memoryless law",
          "[adaptive_controller]") {
  AdaptiveControlConfig cfg = make_adaptive_cfg();
  cfg.controller = StepController::pi;
  cfg.pi_k_i = 1.0;
  cfg.pi_k_p = 0.0;
  cfg.growth_max = 5.0;
  cfg.shrink_max = 0.1;
  cfg.max_dt = 10.0;
  AdaptiveTimeController pi(cfg, 3);
  Time time({0.0, 10.0, 0.1}, 0.0);
  accept_metric(pi, time, 0.5);
  time.begin_attempt(0.1);
  const auto again = pi.decide(0.1, unit_norm(0.2 * 2e-5));
  REQUIRE(again.next_dt == Catch::Approx(0.1 * 0.9 * std::pow(0.2, -1.0 / 3.0)));
}

TEST_CASE("an accepted zero metric clears PI history", "[adaptive_controller]") {
  AdaptiveControlConfig cfg = make_adaptive_cfg();
  cfg.controller = StepController::pi;
  cfg.atol = 1e-5;
  cfg.rtol = 0.0;
  AdaptiveTimeController ctl(cfg, 3);
  Time time({0.0, 10.0, 0.1}, 0.0);
  time.begin_attempt(0.1);
  const auto first = ctl.decide(0.1, unit_norm(0.5 * 1e-5));
  REQUIRE(first.accepted);
  ctl.apply(time, first);
  REQUIRE(ctl.has_previous_accepted_metric());

  time.begin_attempt(0.1);
  const auto zero = ctl.decide(0.1, unit_norm(0.0));
  REQUIRE(zero.accepted);
  REQUIRE(zero.metric == Catch::Approx(0.0));
  REQUIRE(zero.next_dt == Catch::Approx(0.1 * cfg.growth_max));
  ctl.apply(time, zero);
  REQUIRE_FALSE(ctl.has_previous_accepted_metric());
}

TEST_CASE("a frozen error sequence gives a deterministic dt sequence",
          "[adaptive_controller]") {
  const double metrics[] = {0.5, 0.2, 1.5, 0.2, 0.05};
  auto run = [&](StepController kind) {
    AdaptiveControlConfig cfg = make_adaptive_cfg();
    cfg.controller = kind;
    cfg.growth_max = 5.0;
    cfg.shrink_max = 0.1;
    cfg.min_dt = 1e-8;
    cfg.max_dt = 10.0;
    AdaptiveTimeController ctl(cfg, 3);
    Time time({0.0, 100.0, 0.1}, 0.0);
    std::vector<double> next;
    for (double metric : metrics) {
      time.begin_attempt(time.get_dt());
      const auto d = ctl.decide(time.get_attempted_dt(), unit_norm(metric * 2e-5));
      next.push_back(d.next_dt);
      ctl.apply(time, d);
    }
    return next;
  };
  const auto pi_a = run(StepController::pi);
  const auto pi_b = run(StepController::pi);
  const auto memoryless = run(StepController::memoryless);
  REQUIRE(pi_a == pi_b);
  REQUIRE(pi_a.front() == Catch::Approx(memoryless.front()));
  REQUIRE(pi_a[1] != Catch::Approx(memoryless[1]));
}

TEST_CASE("sequential rejections stop at the cap", "[adaptive_controller]") {
  AdaptiveControlConfig cfg = make_adaptive_cfg();
  cfg.max_sequential_rejections = 3;
  AdaptiveTimeController ctl(cfg, 3);
  Time time({0.0, 1.0, 0.1}, 0.0);
  for (int i = 0; i < 2; ++i) {
    time.begin_attempt(0.1);
    const auto d = ctl.decide(0.1, unit_norm(1.0));
    REQUIRE_FALSE(d.accepted);
    ctl.apply(time, d);
  }
  REQUIRE(ctl.sequential_rejections() == 2);
  REQUIRE(time.get_accepted_time() == Catch::Approx(0.0));
  time.begin_attempt(0.1);
  const auto last = ctl.decide(0.1, unit_norm(1.0));
  REQUIRE_THROWS_AS(ctl.apply(time, last), std::runtime_error);
  REQUIRE(ctl.sequential_rejections() == 3);
  REQUIRE(time.get_accepted_time() == Catch::Approx(0.0));
}

TEST_CASE("invalid evidence rejects without writing PI history",
          "[adaptive_controller]") {
  AdaptiveControlConfig cfg = make_adaptive_cfg();
  cfg.controller = StepController::pi;
  AdaptiveTimeController ctl(cfg, 3);
  Time time({0.0, 1.0, 0.1}, 0.0);
  time.begin_attempt(0.1);
  const auto d = ctl.decide(
      0.1, make_invalid_evidence(pfc::integrator::EvidenceKind::EmbeddedPair));
  REQUIRE_FALSE(d.accepted);
  REQUIRE_FALSE(d.decision_available);
  ctl.apply(time, d);
  REQUIRE_FALSE(ctl.has_previous_accepted_metric());
  REQUIRE(time.get_accepted_time() == Catch::Approx(0.0));
  REQUIRE(ctl.rejected_count() == 1);
}

TEST_CASE("restored PI history reproduces the next dt", "[adaptive_controller]") {
  AdaptiveControlConfig cfg = make_adaptive_cfg();
  cfg.controller = StepController::pi;
  cfg.growth_max = 5.0;
  cfg.shrink_max = 0.1;
  cfg.min_dt = 1e-8;
  cfg.max_dt = 10.0;
  cfg.max_sequential_rejections = 4;
  AdaptiveTimeController original(cfg, 3);
  Time time({0.0, 100.0, 0.1}, 0.0);
  accept_metric(original, time, 0.4);
  time.begin_attempt(0.1);
  original.apply(time, original.decide(0.1, unit_norm(1.0)));
  time.begin_attempt(0.1);
  original.apply(time, original.decide(0.1, unit_norm(1.0)));
  REQUIRE(original.sequential_rejections() == 2);

  const AdaptiveControllerState saved = original.capture_state();
  AdaptiveTimeController restored(cfg, 3);
  restored.restore_state(saved);
  const auto from_original = original.decide(0.1, unit_norm(0.3 * 2e-5));
  const auto from_restored = restored.decide(0.1, unit_norm(0.3 * 2e-5));
  REQUIRE(from_restored.next_dt == Catch::Approx(from_original.next_dt));
  REQUIRE(from_restored.accepted);

  restored.reset_history();
  const auto forgotten = restored.decide(0.1, unit_norm(0.3 * 2e-5));
  REQUIRE(forgotten.next_dt == Catch::Approx(0.1 * 0.9 * std::pow(0.3, -1.0 / 3.0)));
  REQUIRE(restored.sequential_rejections() == 2);

  // The streak still counts. Two further rejections reach the cap of 4.
  AdaptiveTimeController capped(cfg, 3);
  capped.restore_state(saved);
  Time cap_time({0.0, 1.0, 0.1}, 0.0);
  cap_time.begin_attempt(0.1);
  capped.apply(cap_time, capped.decide(0.1, unit_norm(1.0)));
  cap_time.begin_attempt(0.1);
  REQUIRE_THROWS_AS(capped.apply(cap_time, capped.decide(0.1, unit_norm(1.0))),
                    std::runtime_error);
}

TEST_CASE("fixed mode ignores and preserves PI history", "[adaptive_controller]") {
  AdaptiveControlConfig cfg;
  cfg.mode = AdaptiveControlMode::fixed;
  AdaptiveTimeController ctl(cfg, 3);
  AdaptiveControllerState saved;
  saved.has_previous_metric = true;
  saved.previous_accepted_metric = 0.2;
  saved.sequential_rejections = 3;
  ctl.restore_state(saved);

  Time time({0.0, 1.0, 0.1}, 0.0);
  time.begin_attempt(0.1);
  const double norms[1] = {1e3};
  const auto ev =
      make_embedded_pair_evidence(norms, AggregationScope::AlreadyReduced, 3);
  const auto d = ctl.decide(0.1, ev);
  REQUIRE(d.accepted);
  REQUIRE(d.next_dt == Catch::Approx(0.1));
  ctl.apply(time, d);
  REQUIRE(ctl.has_previous_accepted_metric());
  REQUIRE(ctl.previous_accepted_metric() == Catch::Approx(0.2));
  REQUIRE(ctl.sequential_rejections() == 0);
}

TEST_CASE("MPI ranks share one embedded metric", "[adaptive_controller][MPI]") {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    return;
  }

  AdaptiveControlConfig cfg = make_adaptive_cfg();
  cfg.atol = 0.0;
  cfg.rtol = 1.0;
  cfg.growth_max = 4.0;
  cfg.shrink_max = 0.25;
  cfg.min_dt = 1e-8;
  cfg.max_dt = 10.0;
  AdaptiveTimeController ctl(cfg, 1);
  // Rank 0 would accept on its own. Rank 1's component dominates.
  const double error = (rank == 0) ? 1e-8 : 2.0;
  const double state = 1.0;
  const double err_v[1] = {error};
  const double state_v[1] = {state};
  const auto d = ctl.decide_from_embedded_error(0.2, err_v, state_v, state_v);
  REQUIRE_FALSE(d.accepted);
  REQUIRE(d.metric == Catch::Approx(2.0));
  REQUIRE(d.next_dt == Catch::Approx(0.2 * 0.9 * 0.5));
}

TEST_CASE("memoryless and PI on a piecewise decay",
          "[adaptive_controller][embedded_rk]") {
  struct Stats {
    int accepted{0};
    int rejected{0};
    unsigned rhs{0};
    double error{0.0};
    bool finished{false};
    std::vector<double> attempted;
    std::vector<int> took;
    std::vector<double> next_dt;
  };
  auto integrate = [](StepController kind) {
    Stats stats;
    AdaptiveControlConfig cfg = make_adaptive_cfg();
    cfg.controller = kind;
    cfg.atol = 1e-5;
    cfg.rtol = 1e-5;
    cfg.max_dt = 0.1;
    cfg.min_dt = 1e-6;
    cfg.max_sequential_rejections = 40;
    AdaptiveTimeController ctl(cfg, 3);
    EmbeddedRKStepper stepper(1, make_embedded_rk23<double>(), PiecewiseStiffRhs{});
    Time time({0.0, 0.5, 0.05}, 0.0);
    std::vector<double> u{1.0};
    int steps = 0;
    while (!time.done() && steps < 800) {
      const double u_before = u[0];
      const double t_before = time.get_accepted_time();
      time.begin_attempt(time.get_dt());
      const double dt = time.get_attempted_dt();
      const auto attempt = stepper.attempt(t_before, dt, u);
      REQUIRE(attempt.success);
      stats.rhs += stepper.last_rhs_evals();
      const auto decision =
          ctl.decide_from_embedded_error(dt, stepper.error(), u, stepper.u_high());
      stats.attempted.push_back(dt);
      stats.took.push_back(decision.accepted ? 1 : 0);
      stats.next_dt.push_back(decision.next_dt);
      if (decision.accepted) {
        commit_step_attempt(u, attempt);
      } else {
        REQUIRE(u[0] == u_before);
      }
      ctl.apply(time, decision);
      if (!decision.accepted) {
        REQUIRE(time.get_accepted_time() == Catch::Approx(t_before));
      }
      ++steps;
    }
    stats.accepted = ctl.accepted_count();
    stats.rejected = ctl.rejected_count();
    stats.finished = time.done();
    stats.error = std::abs(u[0] - std::exp(-7.85));
    return stats;
  };

  const auto memoryless = integrate(StepController::memoryless);
  const auto pi = integrate(StepController::pi);
  const auto pi_again = integrate(StepController::pi);
  REQUIRE(memoryless.finished);
  REQUIRE(pi.finished);
  REQUIRE(memoryless.error < 1e-3);
  REQUIRE(pi.error < 1e-3);
  REQUIRE(memoryless.rhs ==
          static_cast<unsigned>(memoryless.accepted + memoryless.rejected) * 4u);
  REQUIRE(pi.rhs == static_cast<unsigned>(pi.accepted + pi.rejected) * 4u);
  REQUIRE(pi.attempted == pi_again.attempted);
  REQUIRE(pi.next_dt == pi_again.next_dt);
  REQUIRE(pi.took == pi_again.took);

  std::cout << "ode memoryless accepted=" << memoryless.accepted
            << " rejected=" << memoryless.rejected << " rhs=" << memoryless.rhs
            << " error=" << memoryless.error << "\n";
  std::cout << "ode pi accepted=" << pi.accepted << " rejected=" << pi.rejected
            << " rhs=" << pi.rhs << " error=" << pi.error << "\n";
  std::cout << "ode memoryless next_dt";
  for (double h : memoryless.next_dt) {
    std::cout << ' ' << h;
  }
  std::cout << "\node pi next_dt";
  for (double h : pi.next_dt) {
    std::cout << ' ' << h;
  }
  std::cout << '\n';
}
