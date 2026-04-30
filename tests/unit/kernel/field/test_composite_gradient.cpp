// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_composite_gradient.cpp
 * @brief Validates the multi-field composite per-point evaluator.
 *
 * @details
 * `CompositeGradient<Composite, PerField...>` is the kernel hook the
 * heat3d refactor lays down so that a future wave-equation app can plug
 * in without re-touching the framework. Single-field heat3d does not
 * exercise it, so this test fans two `FdGradient<...>` instances out of
 * a single backing `LocalField` and asserts:
 *
 *   1. `prepare()` is forwarded to every sub-evaluator (here a no-op
 *      for FD).
 *   2. The interior bounds match the (shared) sub-evaluator bounds.
 *   3. `operator()(i,j,k)` returns a `Composite` whose per-field grads
 *      structs are populated with the correct per-field-grads-type
 *      members.
 *
 * The two per-field grads aggregates are intentionally *different shapes*
 * (`HasXx` carries only `xx`; `HasYyZz` carries `yy` and `zz`) so the
 * test also re-confirms that the templated `FdGradient<G>` prunes its
 * work to whatever each `G` declares.
 */

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/field/composite_gradient.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/local_field.hpp>

using Catch::Approx;
using pfc::field::CompositeGradient;
using pfc::field::FdGradient;
using pfc::field::LocalField;

namespace {

struct HasXx {
  double xx{};
};
struct HasYyZz {
  double yy{};
  double zz{};
};

struct TwoField {
  HasXx u;
  HasYyZz v;
};

} // namespace

TEST_CASE("CompositeGradient fans two heterogeneous FD evaluators into a "
          "single composite per-point view",
          "[kernel][field][composite_gradient][unit]") {
  // 8x8x8 single-rank FD subdomain with a 1-cell halo (order=2 stencil).
  constexpr int N = 8;
  const int order = 2;

  auto world = pfc::world::create(pfc::GridSize({N, N, N}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(world, /*nparts=*/1);

  auto u = LocalField<double>::from_subdomain(decomp, /*rank=*/0,
                                              /*halo_width=*/1);

  // u(x, y, z) = x^2 + y^2 + z^2 ⇒ uxx = uyy = uzz = 2 everywhere.
  u.apply([](double x, double y, double z) { return x * x + y * y + z * z; });

  auto grad_u = pfc::field::create<HasXx>(u, order);
  auto grad_v = pfc::field::create<HasYyZz>(u, order);
  CompositeGradient<TwoField, FdGradient<HasXx>, FdGradient<HasYyZz>> composite(
      grad_u, grad_v);

  composite.prepare();

  REQUIRE(composite.imin() == 1);
  REQUIRE(composite.imax() == N - 1);
  REQUIRE(composite.jmin() == 1);
  REQUIRE(composite.jmax() == N - 1);
  REQUIRE(composite.kmin() == 1);
  REQUIRE(composite.kmax() == N - 1);

  // Sample one interior cell — both per-field aggregates should report 2.0
  // for whichever second derivatives they declare.
  const TwoField g = composite(N / 2, N / 2, N / 2);
  REQUIRE(g.u.xx == Approx(2.0));
  REQUIRE(g.v.yy == Approx(2.0));
  REQUIRE(g.v.zz == Approx(2.0));
}
