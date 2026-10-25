// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <openpfc/kernel/field/reaction_flux.hpp>
using namespace pfc::field::fd;
using Catch::Approx;
namespace {
auto valid = [](const ReactionPoint &p, double) { return p.state >= 0; };
ReactionPoint point(double u, double t = 0) {
  return {u, {}, {0, 0, 0}, {-1, 0, 0}, t};
}
} // namespace
TEST_CASE("Reaction callback uses local immutable data and physical flux",
          "[unit][field][reaction]") {
  DiffusionAxis axis(17, 1. / 16);
  FluxLedger ledger;
  ledger.begin();
  auto constant = reaction_face(
      point(1), 2., [](const auto &, double k) { return ReactionRate{k}; }, valid,
      ledger);
  FaceCondition prescribed{BoundaryQuantity::ConstitutiveFlux,
                           [](double) { return 2.; }};
  auto u = [](int i) { return 1 + i / 16.; };
  auto d = [](int i) { return 1 + i / 32.; };
  for (int i = 0; i < 17; ++i)
    REQUIRE(axis.apply(i, u, d, 0, &constant) ==
            axis.apply(i, u, d, 0, &prescribed));
  ledger.reject();
  const std::array<double, 1> fields{.4};
  ledger.begin();
  for (double t : {0., .5, 1.}) {
    ReactionPoint p{.7, fields, {.3, 0, 0}, {-1, 0, 0}, t};
    auto face = reaction_face(
        p, 2.,
        [](const auto &v, double k) {
          return ReactionRate{k * v.state * v.state + v.other_fields[0] +
                              v.position[0] * v.outward_normal[0] + v.stage_time};
        },
        valid, ledger);
    double expected = 2 * .7 * .7 + .4 - .3 + t;
    REQUIRE(face.flux(t, 1) == Approx(expected).margin(1e-14));
    double inventory = 0, scale = 1;
    for (int i = 0; i < 17; ++i) {
      double term = axis.weight(i) * axis.apply(i, u, d, t, &face);
      inventory += term;
      scale += std::abs(term);
    }
    REQUIRE(std::abs(inventory + expected) < 1e-11 * scale);
    ledger.stage(face.flux(t, 1), 1, t == .5 ? 2. / 3 : 1. / 6);
  }
  ledger.accept();
  REQUIRE(ledger.accepted() == Approx(2 * .7 * .7 + .4 - .3 + .5));
}
TEST_CASE("Reaction failures poison trial without accepting flux",
          "[unit][field][reaction]") {
  FluxLedger ledger;
  auto law = [](const auto &, double k) { return ReactionRate{k}; };
  REQUIRE_THROWS(reaction_face(point(1), 1., law, valid, ledger));
  for (int failure = 0; failure < 7; ++failure) {
    ledger.begin();
    ledger.stage(1, 1, .1);
    auto p = point(failure == 0 ? -1 : 1);
    if (failure == 1) p.state = std::numeric_limits<double>::quiet_NaN();
    if (failure == 2) p.outward_normal = {1, 1, 0};
    auto failed = [&](const auto &, double) -> ReactionRate {
      if (failure == 3) return {std::numeric_limits<double>::infinity()};
      if (failure == 4) return {0, false};
      if (failure == 5) throw std::runtime_error("local solve exception");
      return {0};
    };
    if (failure == 6) p.stage_time = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS(reaction_face(p, 1., failed, valid, ledger));
    REQUIRE_THROWS(ledger.accept());
    ledger.reject();
    REQUIRE(ledger.accepted() == 0);
  }
}
TEST_CASE("Reversible face exchange relaxes to analytic equilibrium",
          "[unit][field][reaction]") {
  const int n = 17;
  const double h = 1. / 16, alpha = 1., eq = 1., dt = 1e-4;
  DiffusionAxis axis(n, h);
  std::vector<double> u(n), a(n), b(n), c(n), d(n), v(n);
  for (int i = 0; i < n; ++i) u[i] = eq + .1 * std::cos(alpha * (i * h - .5));
  FluxLedger ledger;
  const double k = alpha * std::tan(alpha / 2);
  auto rhs = [&](const std::vector<double> &s, double t, double w,
                 std::vector<double> &out) {
    auto law = [&](const ReactionPoint &p, double rate) {
      return ReactionRate{rate * (p.state - eq)};
    };
    auto lp = point(s.front(), t), rp = point(s.back(), t);
    rp.position = {1, 0, 0};
    rp.outward_normal = {1, 0, 0};
    auto left = reaction_face(lp, k, law, valid, ledger);
    auto right = reaction_face(rp, k, law, valid, ledger);
    for (int i = 0; i < n; ++i)
      out[i] = axis.apply(
          i, [&](int j) { return s[j]; }, [](int) { return 1.; }, t, &left, &right);
    ledger.stage(left.flux(t, 1) + right.flux(t, 1), 1, w);
  };
  double initial = 0;
  for (int i = 0; i < n; ++i) initial += axis.weight(i) * u[i];
  for (int step = 0; step < 100000; ++step) {
    double t = step * dt;
    ledger.begin();
    rhs(u, t, dt / 6, a);
    for (int i = 0; i < n; ++i) v[i] = u[i] + dt * a[i] / 2;
    rhs(v, t + dt / 2, dt / 3, b);
    for (int i = 0; i < n; ++i) v[i] = u[i] + dt * b[i] / 2;
    rhs(v, t + dt / 2, dt / 3, c);
    for (int i = 0; i < n; ++i) v[i] = u[i] + dt * c[i];
    rhs(v, t + dt, dt / 6, d);
    for (int i = 0; i < n; ++i) u[i] += dt * (a[i] + 2 * b[i] + 2 * c[i] + d[i]) / 6;
    ledger.accept();
    if (step == 99 || step == 99999) {
      double error = 0;
      for (int i = 0; i < n; ++i) {
        double exact = eq + .1 * std::exp(-alpha * alpha * (t + dt)) *
                                std::cos(alpha * (i * h - .5));
        error += axis.weight(i) * (u[i] - exact) * (u[i] - exact);
      }
      REQUIRE(std::sqrt(error) <= 1e-4);
    }
  }
  double mass = 0, dist = 0;
  for (int i = 0; i < n; ++i) {
    mass += axis.weight(i) * u[i];
    dist += axis.weight(i) * (u[i] - eq) * (u[i] - eq);
  }
  REQUIRE(std::sqrt(dist) <= 1e-4);
  REQUIRE(std::abs(mass - initial + ledger.accepted()) < 1e-11);
}

TEST_CASE("Weighted reaction sources share the inventory ledger",
          "[unit][field][reaction]") {
  FluxLedger ledger;
  auto p = point(2, .3);
  p.outward_normal = {.6, .8, 0};
  auto law = [](const ReactionPoint &v, double k) {
    return ReactionRate{k * (v.state * v.state - 1) + v.stage_time +
                        v.outward_normal[0]};
  };
  ledger.begin();
  double source = reaction_source(p, 3., .125, .02, 2., law, valid, ledger);
  REQUIRE(source == Approx(-3 * (6 + .3 + .6)));
  ledger.reject();
  REQUIRE(ledger.accepted() == 0);
  ledger.begin();
  source = reaction_source(p, 3., .125, .02, 2., law, valid, ledger);
  ledger.accept();
  REQUIRE(source * .125 * .02 + ledger.accepted() == Approx(0).margin(1e-15));
  ledger.begin();
  REQUIRE(reaction_source(p, 0., .125, .02, 2., law, valid, ledger) == 0);
  ledger.reject();
  for (int failure = 0; failure < 6; ++failure) {
    ledger.begin();
    auto q = p;
    double rho = 3, vol = .125, dt = .02;
    if (failure == 0) rho = -1;
    if (failure == 1) vol = 0;
    if (failure == 2) dt = std::numeric_limits<double>::infinity();
    if (failure == 3) q.outward_normal = {.6, .7, 0};
    if (failure == 4) q.state = -1;
    if (failure == 5) rho = std::numeric_limits<double>::max();
    REQUIRE_THROWS(reaction_source(q, rho, vol, dt, 2., law, valid, ledger));
    REQUIRE_THROWS(ledger.accept());
    ledger.reject();
  }
  ledger.begin();
  REQUIRE_THROWS(reaction_face(p, 2., law, valid, ledger));
  ledger.reject();
}
