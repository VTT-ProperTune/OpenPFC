// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdio>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/field/composite_gradient.hpp>
#include <openpfc/kernel/simulation/steppers/explicit_rk.hpp>

namespace {
using Field = pfc::data::Field<double>;
struct Local {
  double value;
};
struct BoundEvaluator {
  Field &field;
  bool *fail = nullptr;
  bool *resize = nullptr;
  void prepare() {
    // Boundary preparation may write stage ghosts before failing.
    if (field.storage_halo()) field(-1, 0, 0) = 999.0;
    if (resize && *resize) field.vec().clear();
    if (fail && *fail) throw std::runtime_error("forced preparation failure");
  }
  Local operator()(int i, int j, int k) const { return {field(i, j, k)}; }
  std::size_t idx(int i, int j, int k) const {
    return field.idx(i, j, k) - field.idx(0, 0, 0);
  }
  int imin() const { return 0; }
  int imax() const { return field.local_size()[0]; }
  int jmin() const { return 0; }
  int jmax() const { return field.local_size()[1]; }
  int kmin() const { return 0; }
  int kmax() const { return field.local_size()[2]; }
};
struct ScalarModel {
  double rhs(double t, const Local &u) const { return u.value * u.value + t; }
};
struct CoupledLocal {
  Local u, v;
};
struct CoupledModel {
  auto rhs(double t, const CoupledLocal &q) const {
    return std::tuple{q.u.value * q.v.value + t, -q.u.value + 2.0 * t};
  }
};
double scalar_heun(double u, double t, double dt) {
  const double k1 = u * u + t;
  const double predicted = u + dt * k1;
  const double k2 = predicted * predicted + t + dt;
  return u + dt * (k1 + k2) / 2.0;
}
} // namespace

TEST_CASE("Canonical RK field factory evaluates actual nonlinear stage values",
          "[rk_stage_binding]") {
  const auto domain = pfc::domain::create({3, 1, 1});
  Field u(domain,
          pfc::Box3i::from_bounds({0, 0, 0}, {domain.size[0] - 1, domain.size[1] - 1,
                                              domain.size[2] - 1}));
  u.vec() = {0.3, 0.5, 0.7};
  const auto accepted = u.vec();
  BoundEvaluator eval{u};
  ScalarModel model;
  auto stepper = pfc::sim::steppers::create(
      u, eval, model, 0.1, pfc::sim::steppers::make_rk2_heun<double>());
  const auto candidate = stepper.attempt(0.75, u);
  REQUIRE(u.vec() == accepted);
  for (std::size_t i = 0; i < accepted.size(); ++i)
    REQUIRE_THAT(
        candidate.candidate[i],
        Catch::Matchers::WithinAbs(scalar_heun(accepted[i], 0.75, 0.1), 1e-14));
}

TEST_CASE("Canonical RK stages preserve padding and restore on preparation failure",
          "[rk_stage_binding]") {
  const auto domain = pfc::domain::create({2, 1, 1});
  Field u(domain,
          pfc::Box3i::from_bounds({0, 0, 0}, {domain.size[0] - 1, domain.size[1] - 1,
                                              domain.size[2] - 1}),
          1);
  std::fill(u.vec().begin(), u.vec().end(), -777.0);
  u(0, 0, 0) = 0.4;
  u(1, 0, 0) = 0.6;
  auto accepted = u.vec();
  bool fail = false, resize = false;
  BoundEvaluator eval{u, &fail, &resize};
  ScalarModel model;
  auto stepper = pfc::sim::steppers::create(
      u, eval, model, 0.1, pfc::sim::steppers::make_rk2_heun<double>());
  auto candidate = stepper.attempt(0.5, u);
  REQUIRE(u.vec() == accepted);
  for (std::size_t q = 0; q < u.size(); ++q) {
    if (q == u.idx(0, 0, 0) || q == u.idx(1, 0, 0)) {
      REQUIRE_THAT(
          candidate.candidate[q],
          Catch::Matchers::WithinAbs(scalar_heun(accepted[q], 0.5, 0.1), 1e-14));
    } else
      REQUIRE(candidate.candidate[q] == accepted[q]);
  }
  fail = true;
  REQUIRE_THROWS_AS(stepper.attempt(0.5, u), std::runtime_error);
  REQUIRE(u.vec() == accepted);
  fail = false;
  resize = true;
  REQUIRE_THROWS_AS(stepper.attempt(0.5, u), std::invalid_argument);
  REQUIRE(u.vec() == accepted);
}

TEST_CASE("Canonical coupled RK publishes both stage fields simultaneously",
          "[rk_stage_binding]") {
  const auto domain = pfc::domain::create({2, 1, 1});
  Field u(domain,
          pfc::Box3i::from_bounds({0, 0, 0}, {domain.size[0] - 1, domain.size[1] - 1,
                                              domain.size[2] - 1})),
      v(domain,
        pfc::Box3i::from_bounds({0, 0, 0}, {domain.size[0] - 1, domain.size[1] - 1,
                                            domain.size[2] - 1}));
  u.vec() = {0.3, 0.7};
  v.vec() = {0.9, 0.4};
  const auto old_u = u.vec(), old_v = v.vec();
  auto eval = pfc::field::create_composite<CoupledLocal>(BoundEvaluator{u},
                                                         BoundEvaluator{v});
  CoupledModel model;
  auto stepper = pfc::sim::steppers::create(
      std::tie(u, v), eval, model, 0.1, pfc::sim::steppers::make_rk2_heun<double>());
  stepper.step(0.5, u, v);
  for (std::size_t i = 0; i < u.size(); ++i) {
    const double ku = old_u[i] * old_v[i] + 0.5, kv = -old_u[i] + 1.0;
    const double pu = old_u[i] + 0.1 * ku, pv = old_v[i] + 0.1 * kv;
    REQUIRE_THAT(u.vec()[i], Catch::Matchers::WithinAbs(
                                 old_u[i] + 0.05 * (ku + pu * pv + 0.6), 1e-14));
    REQUIRE_THAT(v.vec()[i], Catch::Matchers::WithinAbs(
                                 old_v[i] + 0.05 * (kv - pu + 1.2), 1e-14));
  }
}

TEST_CASE("Canonical RK rejects changed or aliased storage before evaluation",
          "[rk_stage_binding]") {
  const auto domain = pfc::domain::create({2, 1, 1});
  Field u(domain,
          pfc::Box3i::from_bounds({0, 0, 0}, {domain.size[0] - 1, domain.size[1] - 1,
                                              domain.size[2] - 1}));
  u.vec() = {0.4, 0.8};
  BoundEvaluator eval{u};
  ScalarModel model;
  auto stepper = pfc::sim::steppers::create(
      u, eval, model, 0.1, pfc::sim::steppers::make_rk2_heun<double>());
  std::vector<double> wrong(3, 9.0);
  REQUIRE_THROWS_AS(stepper.attempt(0.0, wrong), std::invalid_argument);
  REQUIRE((u.vec() == std::vector<double>{0.4, 0.8}));
  std::vector<double> replacement{0.2, 0.9};
  u.vec().swap(replacement);
  const auto changed = u.vec();
  REQUIRE_THROWS_AS(stepper.attempt(0.0, u), std::invalid_argument);
  REQUIRE(u.vec() == changed);
  auto both = pfc::field::create_composite<CoupledLocal>(BoundEvaluator{u},
                                                         BoundEvaluator{u});
  CoupledModel coupled;
  REQUIRE_THROWS_AS(
      pfc::sim::steppers::create(std::tie(u, u), both, coupled, 0.1,
                                 pfc::sim::steppers::make_rk2_heun<double>()),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::sim::steppers::create(eval, model, 0.1, u.size(),
                                 pfc::sim::steppers::make_rk2_heun<double>()),
      std::invalid_argument);
}

TEST_CASE("Canonical scalar RK factories attain midpoint and RK4 temporal order",
          "[rk_stage_binding]") {
  struct ManufacturedModel {
    double rhs(double t, const Local &q) const {
      return q.value * q.value + std::exp(t) - std::exp(2.0 * t);
    }
  } model;
  for (const int order : {2, 4}) {
    auto error = [&](double dt) {
      const auto domain = pfc::domain::create({1, 1, 1});
      Field u(domain, pfc::Box3i::from_bounds({0, 0, 0}, {0, 0, 0}));
      u.vec()[0] = std::exp(0.2);
      BoundEvaluator eval{u};
      const auto tableau = order == 2
                               ? pfc::sim::steppers::make_rk2_midpoint<double>()
                               : pfc::sim::steppers::make_rk4_classical<double>();
      auto stepper = pfc::sim::steppers::create(u, eval, model, dt, tableau);
      const int steps = std::lround(0.2 / dt);
      for (int n = 0; n < steps; ++n) stepper.step(0.2 + n * dt, u);
      return std::abs(u.vec()[0] - std::exp(0.4));
    };
    const double coarse = error(0.05), fine = error(0.025);
    const double rate = std::log2(coarse / fine);
    std::fprintf(stderr, "RK%d scalar temporal errors %.17g %.17g rate %.9g\n",
                 order, coarse, fine, rate);
    INFO("order=" << order << " coarse=" << coarse << " fine=" << fine
                  << " rate=" << rate);
    REQUIRE(rate > order - 0.3);
    REQUIRE(rate < order + 0.3);
  }
}

TEST_CASE("RK fixed scratch rejects malformed RHS buffers without accepted mutation",
          "[rk_stage_binding]") {
  using namespace pfc::sim::steppers;
  std::vector<double> u(2, 1.0), v(2, 2.0);
  auto scalar_rhs = [](double, auto &, auto &du) { du.clear(); };
  ExplicitRKStepper scalar(0.1, 2, make_rk2_midpoint<double>(), scalar_rhs);
  REQUIRE_THROWS_AS(scalar.attempt(0.0, u), std::invalid_argument);
  REQUIRE_THROWS_AS(scalar.attempt(0.0, u), std::invalid_argument);
  REQUIRE((u == std::vector<double>{1.0, 1.0}));
  auto multi_rhs = [](double, auto &stage, auto &) { std::get<0>(stage).clear(); };
  MultiExplicitRKStepper<decltype(multi_rhs), 2> multi(
      0.1, {2, 2}, make_rk2_midpoint<double>(), multi_rhs);
  REQUIRE_THROWS_AS(multi.step(0.0, u, v), std::invalid_argument);
  REQUIRE_THROWS_AS(multi.step(0.0, u, v), std::invalid_argument);
  REQUIRE((u == std::vector<double>{1.0, 1.0}));
  REQUIRE((v == std::vector<double>{2.0, 2.0}));
}

namespace {
struct TimedLocal {
  double u, v, w, forcing;
};
struct TimedEvaluator {
  Field &u, &v, &w;
  std::vector<double> times;
  double forcing = -999;
  bool fail = false;
  // Intentionally no prepare(): the timed hook is a standalone contract.
  void prepare(double t) {
    times.push_back(t);
    forcing = t;
    if (fail) throw std::runtime_error("timed preparation failure");
  }
  TimedLocal operator()(int i, int j, int k) const {
    return {u(i, j, k), v(i, j, k), w(i, j, k), forcing};
  }
  std::size_t idx(int i, int j, int k) const {
    return u.idx(i, j, k) - u.idx(0, 0, 0);
  }
  int imin() const { return 0; }
  int imax() const { return u.local_size()[0]; }
  int jmin() const { return 0; }
  int jmax() const { return 1; }
  int kmin() const { return 0; }
  int kmax() const { return 1; }
};
struct TimedModel {
  auto rhs(double, const TimedLocal &g) const {
    // A nonautonomous coupled invariant with forcing cached in preparation.
    const double a = g.forcing * g.u, b = g.forcing * g.v;
    return std::tuple{a, b, -a - b};
  }
};
} // namespace
TEST_CASE("Coupled RK preparation receives actual stage time",
          "[rk_stage_binding]") {
  const auto domain = pfc::domain::create({2, 1, 1});
  const auto box = pfc::Box3i::from_bounds({0, 0, 0}, {1, 0, 0});
  Field u(domain, box, 1), v(domain, box, 1), w(domain, box, 1);
  u(0, 0, 0) = 1;
  u(1, 0, 0) = 2;
  v(0, 0, 0) = 3;
  v(1, 0, 0) = 4;
  w(0, 0, 0) = -4;
  w(1, 0, 0) = -6;
  const auto old_u = u.vec(), old_v = v.vec(), old_w = w.vec();
  TimedEvaluator eval{u, v, w, {}, -999, false};
  TimedModel model;
  auto stepper =
      pfc::sim::steppers::create(std::tie(u, v, w), eval, model, 0.1,
                                 pfc::sim::steppers::make_rk4_classical<double>(),
                                 pfc::sim::LoopPolicy::Serial);
  stepper.step(0.5, u, v, w);
  REQUIRE(eval.times == std::vector<double>{0.5, 0.55, 0.55, 0.6});
  // Independent scalar RK4 expansion for y'=t y.
  const double k1 = 0.5, k2 = 0.55 * (1 + 0.05 * k1);
  const double k3 = 0.55 * (1 + 0.05 * k2), k4 = 0.6 * (1 + 0.1 * k3);
  const double gain = 1 + 0.1 * (k1 + 2 * k2 + 2 * k3 + k4) / 6;
  for (int i = 0; i < 2; ++i) {
    const auto q = u.idx(i, 0, 0);
    REQUIRE_THAT(u.vec()[q], Catch::Matchers::WithinAbs(old_u[q] * gain, 1e-14));
    REQUIRE_THAT(u.vec()[q] + v.vec()[q] + w.vec()[q],
                 Catch::Matchers::WithinAbs(0, 1e-14));
  }
  const auto accepted_u = u.vec(), accepted_v = v.vec(), accepted_w = w.vec();
  eval.fail = true;
  REQUIRE_THROWS_AS(stepper.step(0.6, u, v, w), std::runtime_error);
  REQUIRE(u.vec() == accepted_u);
  REQUIRE(v.vec() == accepted_v);
  REQUIRE(w.vec() == accepted_w);
}

TEST_CASE("Coupled RK final checks restore accepted fields and allow retry",
          "[rk_stage_binding]") {
  const auto geometry = pfc::domain::create({2, 3, 4});
  const auto box = pfc::Box3i::from_bounds({0, 0, 0}, {1, 2, 3});
  struct Oscillator {
    auto rhs(double, const CoupledLocal &q) const {
      return std::tuple{q.v.value, -q.u.value};
    }
  } model;
  for (bool fourth : {false, true}) {
    Field u(geometry, box, 1), v(geometry, box, 1);
    std::fill(u.vec().begin(), u.vec().end(), -777.0);
    std::fill(v.vec().begin(), v.vec().end(), -333.0);
    for (int k = 0; k < 4; ++k)
      for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 2; ++i) {
          u(i, j, k) = .3 + .1 * i + .05 * j + .02 * k;
          v(i, j, k) = .9 - .2 * i + .03 * j - .04 * k;
        }
    const auto before_u = u.vec(), before_v = v.vec();
    const auto *address_u = u.data(), *address_v = v.data();
    bool stage_failure = false;
    auto eval = pfc::field::create_composite<CoupledLocal>(
        BoundEvaluator{u, &stage_failure}, BoundEvaluator{v});
    const auto tableau = fourth ? pfc::sim::steppers::make_rk4_classical<double>()
                                : pfc::sim::steppers::make_rk2_heun<double>();
    auto rk = pfc::sim::steppers::create(std::tie(u, v), eval, model, .1, tableau,
                                         pfc::sim::LoopPolicy::Serial);
    const double dt = .1;
    const double a = 1 - dt * dt / 2 + (fourth ? dt * dt * dt * dt / 24 : 0);
    const double b = dt - (fourth ? dt * dt * dt / 6 : 0);
    int calls = 0;
    bool reject = true;
    auto final_check = [&](double proposed_time) {
      ++calls;
      REQUIRE_THAT(proposed_time, Catch::Matchers::WithinAbs(.8, 1e-15));
      REQUIRE(u.data() == address_u);
      REQUIRE(v.data() == address_v);
      for (int k = 0; k < 4; ++k)
        for (int j = 0; j < 3; ++j)
          for (int i = 0; i < 2; ++i) {
            const auto q = u.idx(i, j, k);
            REQUIRE_THAT(u(i, j, k), Catch::Matchers::WithinAbs(
                                         a * before_u[q] + b * before_v[q], 1e-14));
            REQUIRE_THAT(v(i, j, k), Catch::Matchers::WithinAbs(
                                         a * before_v[q] - b * before_u[q], 1e-14));
          }
      if (reject) throw std::runtime_error("forced final rejection");
    };
    REQUIRE_THROWS_WITH(rk.step_checked(.7, final_check, u, v),
                        "forced final rejection");
    REQUIRE(calls == 1);
    REQUIRE(u.vec() == before_u);
    REQUIRE(v.vec() == before_v);
    REQUIRE(u.data() == address_u);
    REQUIRE(v.data() == address_v);
    reject = false;
    REQUIRE_THAT(rk.step_checked(.7, final_check, u, v),
                 Catch::Matchers::WithinAbs(.8, 1e-15));
    REQUIRE(calls == 2);
    const auto accepted_u = u.vec(), accepted_v = v.vec();
    stage_failure = true;
    REQUIRE_THROWS_AS(rk.step_checked(.8, final_check, u, v), std::runtime_error);
    REQUIRE(calls == 2);
    REQUIRE(u.vec() == accepted_u);
    REQUIRE(v.vec() == accepted_v);
  }
}
