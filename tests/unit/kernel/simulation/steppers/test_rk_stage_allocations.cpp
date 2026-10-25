// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "fixtures/allocation_provider.hpp"
#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/field/composite_gradient.hpp>
#include <openpfc/kernel/simulation/steppers/explicit_rk.hpp>

namespace {
struct Local {
  double value;
};
struct Pair {
  Local u, v;
};
struct Eval {
  pfc::data::Field<double> &u;
  void prepare() {}
  Local operator()(int i, int j, int k) const { return {u(i, j, k)}; }
  std::size_t idx(int i, int j, int k) const { return u.idx(i, j, k); }
  int imin() const { return 0; }
  int imax() const { return u.local_size()[0]; }
  int jmin() const { return 0; }
  int jmax() const { return 1; }
  int kmin() const { return 0; }
  int kmax() const { return 1; }
};
struct SingleModel {
  double rhs(double t, const Local &q) const { return -q.value + t; }
};
struct PairModel {
  auto rhs(double t, const Pair &q) const {
    return std::tuple{q.v.value + t, -q.u.value};
  }
};
} // namespace

TEST_CASE("Warmed canonical scalar and multi RK4 steps allocate no host scratch",
          "[rk_stage_allocations]") {
  const auto domain = pfc::domain::create({8, 1, 1});
  const auto box = pfc::Box3i::from_bounds({0, 0, 0}, {7, 0, 0});
  pfc::data::Field<double> u(domain, box), v(domain, box);
  std::fill(u.vec().begin(), u.vec().end(), 0.25);
  std::fill(v.vec().begin(), v.vec().end(), 0.5);
  Eval eval{u};
  SingleModel single;
  auto pair_eval = pfc::field::create_composite<Pair>(Eval{u}, Eval{v});
  PairModel pair;
  const auto tableau = pfc::sim::steppers::make_rk4_classical<double>();
  auto scalar = pfc::sim::steppers::create(u, eval, single, 0.001, tableau);
  auto multi =
      pfc::sim::steppers::create(std::tie(u, v), pair_eval, pair, 0.001, tableau);
  scalar.step(0.0, u);
  multi.step(0.0, u, v);
  pfc::diagnostics::Traffic traffic;
  {
    pfc::diagnostics::Scope scope(&traffic);
    for (int n = 0; n < 8; ++n) {
      scalar.step(0.001 * n, u);
      multi.step(0.001 * n, u, v);
    }
  }
  REQUIRE(traffic.allocations(pfc::diagnostics::Space::Host).requests == 0);
}
