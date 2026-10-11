// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <catch2/catch_test_macros.hpp>
#include <openpfc/solvers/newton.hpp>
using namespace pfc::solvers;
namespace {
struct SquareRoot {
  double at = 0, cache = 0, saved = 0;
  int committed = 27, accepted = 0, rejected = 0;
  void residual(const std::vector<double> &x, std::vector<double> &r) {
    cache = x[0];
    r = {x[0] * x[0] - 2};
  }
  void linearize(const std::vector<double> &x) { at = x[0]; }
  void jacobian(const std::vector<double> &v, std::vector<double> &w) {
    w = {2 * at * v[0]};
  }
  void precondition(const std::vector<double> &r, std::vector<double> &z, int) {
    z = r;
  }
  void begin_trial(const std::vector<double> &) { saved = cache; }
  void accept_trial(const std::vector<double> &) { ++accepted; }
  void reject_trial() {
    cache = saved;
    ++rejected;
  }
};
struct Coupled {
  std::vector<double> at;
  void residual(const std::vector<double> &x, std::vector<double> &r) {
    r = {std::atan(x[0]) + .2 * x[1], x[1] + x[1] * x[1] * x[1] - .5 * x[0]};
  }
  void linearize(const std::vector<double> &x) { at = x; }
  void jacobian(const std::vector<double> &v, std::vector<double> &w) {
    w = {v[0] / (1 + at[0] * at[0]) + .2 * v[1],
         -.5 * v[0] + (1 + 3 * at[1] * at[1]) * v[1]};
  }
  void precondition(const std::vector<double> &r, std::vector<double> &z, int) {
    z = r;
  }
  void begin_trial(const std::vector<double> &) {}
  void accept_trial(const std::vector<double> &) {}
  void reject_trial() {}
};
} // namespace
TEST_CASE("Newton backtracks while preserving permanent state", "[nonlinear]") {
  VectorOperations ops;
  SquareRoot model;
  std::vector<double> x{.1};
  NewtonControls c;
  c.relative_tolerance = 1e-12;
  auto result = newton(ops, model, x, c);
  REQUIRE(result.converged());
  REQUIRE(result.rejected_trials > 0);
  REQUIRE(model.rejected == result.rejected_trials);
  REQUIRE(model.committed == 27);
  REQUIRE(std::abs(x[0] - std::sqrt(2.)) < 1e-11);
  REQUIRE(std::abs(model.cache - x[0]) < 1e-15);
  for (const auto &s : result.history) {
    REQUIRE(s.linear.converged());
    REQUIRE(s.linear.residual <= s.forcing * s.residual_before);
    REQUIRE(s.residual_after <=
            (1 - c.armijo * s.step_length * (1 - s.forcing)) * s.residual_before);
  }
}
TEST_CASE("failed line search restores accepted iterate and cache", "[nonlinear]") {
  VectorOperations ops;
  SquareRoot model;
  std::vector<double> x{.1};
  NewtonControls c;
  c.maximum_backtracks = 0;
  auto result = newton(ops, model, x, c);
  REQUIRE(result.status == NonlinearStatus::LineSearchFailure);
  REQUIRE(x[0] == .1);
  REQUIRE(model.cache == .1);
  REQUIRE(model.accepted == 0);
  REQUIRE(model.rejected == 1);
  REQUIRE(model.committed == 27);
  c.maximum_iterations = 0;
  REQUIRE(newton(ops, model, x, c).status == NonlinearStatus::IterationLimit);
}
TEST_CASE("adaptive and tight Newton reach an independent coupled root",
          "[nonlinear]") {
  VectorOperations ops;
  Coupled model;
  std::vector<double> tight{2, .5}, adaptive = tight;
  NewtonControls c;
  c.relative_tolerance = 1e-11;
  c.maximum_iterations = 40;
  c.adaptive_forcing = false;
  c.initial_forcing = 1e-9;
  auto fixed = newton(ops, model, tight, c);
  REQUIRE(fixed.converged());
  c.adaptive_forcing = true;
  c.initial_forcing = .1;
  auto result = newton(ops, model, adaptive, c);
  REQUIRE(result.converged());
  REQUIRE(ops.norm(tight) < 1e-10);
  REQUIRE(ops.norm(adaptive) < 1e-10);
  bool changed = false;
  for (std::size_t k = 1; k < result.history.size(); ++k) {
    const auto &previous = result.history[k - 1];
    double eta = c.forcing_gamma *
                 std::pow(previous.residual_after / previous.residual_before,
                          c.forcing_power);
    double safeguard = c.forcing_gamma * std::pow(previous.forcing, c.forcing_power);
    if (safeguard > .1) eta = std::max(eta, safeguard);
    eta = std::clamp(eta, c.minimum_forcing, c.maximum_forcing);
    REQUIRE(std::abs(result.history[k].forcing - eta) < 1e-15);
    changed = changed || eta != c.initial_forcing;
  }
  REQUIRE(changed);
}
