// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/field/composite_gradient.hpp>
#include <openpfc/kernel/simulation/steppers/explicit_rk.hpp>

namespace {

struct Local {
  double value{};
};
struct Triple {
  Local u, v, w;
};
struct Quad {
  Local u, v, w, q;
};
struct Eval {
  pfc::data::Field<double> &f;
  void prepare() {}
  Local operator()(int i, int j, int k) const { return {f(i, j, k)}; }
  std::size_t idx(int i, int j, int k) const { return f.idx(i, j, k); }
  int imin() const { return 0; }
  int imax() const { return f.local_size()[0]; }
  int jmin() const { return 0; }
  int jmax() const { return 1; }
  int kmin() const { return 0; }
  int kmax() const { return 1; }
};
struct TripleModel {
  auto rhs(double t, const Triple &g) const {
    return std::tuple{g.v.value + t, g.w.value, -g.u.value};
  }
};
struct QuadModel {
  auto rhs(double t, const Quad &g) const {
    return std::tuple{g.v.value + t, g.w.value, g.q.value, -g.u.value};
  }
};

pfc::data::Field<double> make_line() {
  const auto domain = pfc::domain::create({8, 1, 1});
  const auto box = pfc::Box3i::from_bounds({0, 0, 0}, {7, 0, 0});
  return pfc::data::Field<double>(domain, box);
}

} // namespace

TEST_CASE("host multi-field RK4 advances three coupled real fields",
          "[rk_nfield][host]") {
  auto u = make_line(), v = make_line(), w = make_line();
  std::fill(u.vec().begin(), u.vec().end(), 0.25);
  std::fill(v.vec().begin(), v.vec().end(), 0.50);
  std::fill(w.vec().begin(), w.vec().end(), 0.75);
  auto eval = pfc::field::create_composite<Triple>(Eval{u}, Eval{v}, Eval{w});
  TripleModel model;
  const auto tableau = pfc::sim::steppers::make_rk4_classical<double>();
  auto rk =
      pfc::sim::steppers::create(std::tie(u, v, w), eval, model, 0.01, tableau);
  const double u0 = u(0, 0, 0);
  rk.step(0.0, u, v, w);
  REQUIRE(u(0, 0, 0) != Catch::Approx(u0));
  REQUIRE(std::isfinite(u(0, 0, 0)));
  REQUIRE(std::isfinite(v(0, 0, 0)));
  REQUIRE(std::isfinite(w(0, 0, 0)));
}

TEST_CASE("host multi-field RK4 advances four coupled real fields",
          "[rk_nfield][host]") {
  auto u = make_line(), v = make_line(), w = make_line(), q = make_line();
  std::fill(u.vec().begin(), u.vec().end(), 0.20);
  std::fill(v.vec().begin(), v.vec().end(), 0.40);
  std::fill(w.vec().begin(), w.vec().end(), 0.60);
  std::fill(q.vec().begin(), q.vec().end(), 0.80);
  auto eval = pfc::field::create_composite<Quad>(Eval{u}, Eval{v}, Eval{w}, Eval{q});
  QuadModel model;
  const auto tableau = pfc::sim::steppers::make_rk4_classical<double>();
  auto rk =
      pfc::sim::steppers::create(std::tie(u, v, w, q), eval, model, 0.01, tableau);
  const double u0 = u(0, 0, 0);
  rk.step(0.0, u, v, w, q);
  REQUIRE(u(0, 0, 0) != Catch::Approx(u0));
  REQUIRE(std::isfinite(q(0, 0, 0)));
}
