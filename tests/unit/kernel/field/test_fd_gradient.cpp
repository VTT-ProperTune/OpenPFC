// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_fd_gradient.cpp
 * @brief Unit tests for `pfc::field::FdGradient<G>` covering the first- and
 *        second-derivative branches and the constructor's diagnostic
 *        rejection of unsupported orders.
 *
 * @details
 * Complements `test_fd_apply.cpp` (which covers the underlying primitives
 * directly) and `test_composite_gradient.cpp` (which already exercises the
 * `xx / yy / zz` path through the composite evaluator). This file adds:
 *
 *  1. `g.x / g.y / g.z` correctness against a linear field on a small
 *     single-rank `LocalField` for orders 2, 4, 6, 8.
 *  2. Mixed `value + x + xx` aggregate on a quadratic field — proves the
 *     D1 and D2 codepaths coexist and the value member is read directly.
 *  3. Constructor throws `std::invalid_argument` when a model declares
 *     `g.x` and asks for an order with no D1 table (16, 18, 20).
 *  4. Constructor throws when a model declares `g.xx` and asks for an
 *     order with no D2 table (22).
 *
 * The `xy / xz / yz` rejection is a `static_assert` so it is verified by
 * the compile success of every other case (any leakage would be caught
 * by a different cell in the matrix).
 */

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/local_field.hpp>

using Catch::Approx;
using pfc::field::LocalField;

namespace {

struct GradXYZ {
  double x{};
  double y{};
  double z{};
};

struct ValueXXX {
  double value{};
  double x{};
  double xx{};
};

struct OnlyXX {
  double xx{};
};

struct OnlyX {
  double x{};
};

LocalField<double> make_field(int N, int hw) {
  auto world = pfc::world::create(pfc::GridSize({N, N, N}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(world, /*nparts=*/1);
  return LocalField<double>::from_subdomain(decomp, /*rank=*/0, hw);
}

} // namespace

TEST_CASE("FdGradient<GradXYZ> recovers exact first derivatives on a linear field",
          "[kernel][field][fd_gradient][unit]") {
  // u(x, y, z) = 2 x + 3 y + 4 z ⇒ u_x = 2, u_y = 3, u_z = 4 everywhere.
  // Central D1 of any order is **exact** for linear functions — every
  // tabulated order should reproduce the analytic derivative bit-for-bit
  // (modulo round-off from the integer-weight summation).
  bool gradients_match = true;
  for (int order : {2, 4, 6, 8}) {
    INFO("order = " << order);
    const int hw = order / 2;
    const int N = order + 4; // small enough that the test is cheap, big
                             // enough that the centre cell is well-inside
                             // the [hw, N-hw) window for orders up to 8.
    auto u = make_field(N, hw);
    u.apply(
        [](double x, double y, double z) { return 2.0 * x + 3.0 * y + 4.0 * z; });

    auto grad = pfc::field::create<GradXYZ>(u, order);
    const GradXYZ g = grad(N / 2, N / 2, N / 2);
    gradients_match &=
        g.x == Approx(2.0) && g.y == Approx(3.0) && g.z == Approx(4.0);
  }
  REQUIRE(gradients_match);
}

TEST_CASE("FdGradient<ValueXXX> populates value, x, and xx coherently",
          "[kernel][field][fd_gradient][unit]") {
  // u(x, y, z) = x^2 ⇒ value(N/2, .) = (N/2)^2; u_x = 2 (N/2); u_xx = 2.
  // Mixing D1 and D2 in one aggregate exercises both stencil tables.
  const int order = 4;
  const int hw = order / 2;
  const int N = 10;
  auto u = make_field(N, hw);
  u.apply([](double x, double /*y*/, double /*z*/) { return x * x; });

  auto grad = pfc::field::create<ValueXXX>(u, order);
  const ValueXXX g = grad(N / 2, N / 2, N / 2);
  REQUIRE(g.value == Approx(static_cast<double>(N / 2) * (N / 2)));
  REQUIRE(g.x == Approx(2.0 * (N / 2)));
  REQUIRE(g.xx == Approx(2.0));
}

TEST_CASE("FdGradient ctor throws when a model needs a missing D1 table",
          "[kernel][field][fd_gradient][unit]") {
  // D1 tables cover orders 2..14; orders 16/18/20 are not yet tabulated.
  // A model that declares `g.x` and asks for any of those should reject
  // construction with a clear `std::invalid_argument` rather than
  // silently producing zeros at runtime.
  for (int order : {16, 18, 20}) {
    const int hw = order / 2;
    const int N = order + 4;
    auto u = make_field(N, hw);
    REQUIRE_THROWS_AS(pfc::field::create<OnlyX>(u, order), std::invalid_argument);
  }
}

TEST_CASE("FdGradient ctor throws when a model needs a missing D2 table",
          "[kernel][field][fd_gradient][unit]") {
  // D2 tables cover orders 2..20; order 22 is not tabulated.
  const int order = 22;
  const int hw = order / 2;
  const int N = order + 4;
  auto u = make_field(N, hw);
  REQUIRE_THROWS_AS(pfc::field::create<OnlyXX>(u, order), std::invalid_argument);
}

TEST_CASE("FdGradient<OnlyX>: order-2 D1 is exact on a constant field "
          "(anti-symmetry) and on a linear ramp (closed form)",
          "[kernel][field][fd_gradient][unit]") {
  const int order = 2;
  const int hw = 1;
  const int N = 8;

  // Constant field ⇒ every D1 result is zero.
  {
    auto u = make_field(N, hw);
    u.apply([](double, double, double) { return 7.5; });
    auto grad = pfc::field::create<OnlyX>(u, order);
    REQUIRE(grad(N / 2, N / 2, N / 2).x == Approx(0.0));
  }

  // Linear ramp u = -3 x ⇒ u_x = -3.
  {
    auto u = make_field(N, hw);
    u.apply([](double x, double, double) { return -3.0 * x; });
    auto grad = pfc::field::create<OnlyX>(u, order);
    REQUIRE(grad(N / 2, N / 2, N / 2).x == Approx(-3.0));
  }
}

TEST_CASE("pfc::gradient::FDGradient binds to a PaddedBrick and matches the "
          "LocalField path",
          "[kernel][field][fd_gradient][unit]") {
  // Same -3 x linear ramp as the OnlyX case above, this time evaluated on a
  // PaddedBrick via the brick-binding constructor (default order = 2). The
  // pfc::field::FdGradient alias must remain spell-compatible.
  using namespace pfc;
  auto world = world::create(GridSize({8, 8, 8}));
  auto decomp = decomposition::create(world, 1);

  field::PaddedBrick<double> u(decomp, /*rank=*/0, /*hw=*/1);
  u.apply([](double x, double, double) { return -3.0 * x; });

  // Default order, free `prepare` is a no-op for FD.
  pfc::gradient::FDGradient<OnlyX> grad(u);
  pfc::gradient::prepare(grad);

  // Free `evaluate` over a centre-cell Int3.
  const pfc::Int3 c{u.nx() / 2, u.ny() / 2, u.nz() / 2};
  const auto g_free = pfc::gradient::evaluate(grad, c);
  REQUIRE(g_free.x == Approx(-3.0));

  // Deprecated alias still resolves to the canonical evaluator.
  static_assert(std::is_same_v<pfc::field::FdGradient<OnlyX>,
                               pfc::gradient::FDGradient<OnlyX>>,
                "pfc::field::FdGradient should alias pfc::gradient::FDGradient");
}

TEST_CASE("FDGradient prepare invokes callback when provided",
          "[kernel][field][fd_gradient][unit]") {
  // Test that prepare() calls the callback when one is provided.
  int callback_invoked = 0;
  auto callback = [&callback_invoked]() { ++callback_invoked; };

  const int order = 2;
  const int hw = order / 2;
  const int N = 8;
  auto u = make_field(N, hw);
  u.apply([](double, double, double) { return 1.0; });

  // Create FDGradient with callback via raw-pointer constructor
  auto grad = pfc::gradient::FDGradient<OnlyX>(
      u.data(), u.size3()[0], u.size3()[1], u.size3()[2], u.spacing()[0],
      u.spacing()[1], u.spacing()[2], u.halo_width(), order, callback);

  REQUIRE(callback_invoked == 0);
  grad.prepare();
  REQUIRE(callback_invoked == 1);
  grad.prepare();
  REQUIRE(callback_invoked == 2);
}

TEST_CASE("FDGradient prepare is no-op without callback",
          "[kernel][field][fd_gradient][unit]") {
  // Test backward compatibility: prepare() does nothing when no callback is
  // provided.
  const int order = 2;
  const int hw = order / 2;
  const int N = 8;
  auto u = make_field(N, hw);
  u.apply([](double, double, double) { return 1.0; });

  // Create FDGradient without callback (default parameter)
  auto grad = pfc::field::create<OnlyX>(u, order);

  // Should not throw and should have no side effects
  REQUIRE_NOTHROW(grad.prepare());
  const auto g1 = grad(N / 2, N / 2, N / 2);
  grad.prepare();
  const auto g2 = grad(N / 2, N / 2, N / 2);
  REQUIRE(g1.x == g2.x);
}

TEST_CASE("FDGradient PaddedBrick constructor accepts callback",
          "[kernel][field][fd_gradient][unit]") {
  // Test that the PaddedBrick constructor accepts and stores the callback.
  int callback_invoked = 0;
  auto callback = [&callback_invoked]() { ++callback_invoked; };

  using namespace pfc;
  auto world = world::create(GridSize({8, 8, 8}));
  auto decomp = decomposition::create(world, 1);

  field::PaddedBrick<double> u(decomp, /*rank=*/0, /*hw=*/1);
  u.apply([](double x, double, double) { return -3.0 * x; });

  // Create FDGradient with callback via PaddedBrick constructor
  pfc::gradient::FDGradient<OnlyX> grad(u, 2, callback);

  REQUIRE(callback_invoked == 0);
  grad.prepare();
  REQUIRE(callback_invoked == 1);

  // Verify gradient computation still works
  const pfc::Int3 c{u.nx() / 2, u.ny() / 2, u.nz() / 2};
  const auto g = pfc::gradient::evaluate(grad, c);
  REQUIRE(g.x == Approx(-3.0));
}

TEST_CASE("FDGradient raw-pointer constructor accepts callback",
          "[kernel][field][fd_gradient][unit]") {
  // Test that the raw-pointer constructor accepts and stores the callback.
  int callback_invoked = 0;
  auto callback = [&callback_invoked]() { ++callback_invoked; };

  const int order = 4;
  const int hw = order / 2;
  const int N = 10;
  auto u = make_field(N, hw);
  u.apply([](double x, double, double) { return 2.0 * x; });

  // Create FDGradient with callback via raw-pointer constructor
  const auto sz = u.size3();
  const auto sp = u.spacing();
  auto grad =
      pfc::gradient::FDGradient<OnlyX>(u.data(), sz[0], sz[1], sz[2], sp[0], sp[1],
                                       sp[2], u.halo_width(), order, callback);

  REQUIRE(callback_invoked == 0);
  grad.prepare();
  REQUIRE(callback_invoked == 1);

  // Verify gradient computation still works
  const auto g = grad(N / 2, N / 2, N / 2);
  REQUIRE(g.x == Approx(2.0));
}

TEST_CASE("halo_preparation calls callback", "[kernel][field][fd_gradient][unit]") {
  int callback_invoked = 0;
  auto callback = [&callback_invoked]() { ++callback_invoked; };

  const int order = 2;
  const int hw = order / 2;
  const int N = 8;
  auto u = make_field(N, hw);
  u.apply([](double, double, double) { return 1.0; });

  auto grad = pfc::gradient::FDGradient<OnlyX>(
      u.data(), u.size3()[0], u.size3()[1], u.size3()[2], u.spacing()[0],
      u.spacing()[1], u.spacing()[2], u.halo_width(), order, callback);

  REQUIRE(callback_invoked == 0);
  grad.halo_preparation();
  REQUIRE(callback_invoked == 1);
  grad.halo_preparation();
  REQUIRE(callback_invoked == 2);
}

TEST_CASE("halo_preparation and prepare identity",
          "[kernel][field][fd_gradient][unit]") {
  int callback_invoked = 0;
  auto callback = [&callback_invoked]() { ++callback_invoked; };

  const int order = 2;
  const int hw = order / 2;
  const int N = 8;
  auto u = make_field(N, hw);
  u.apply([](double, double, double) { return 1.0; });

  auto grad = pfc::gradient::FDGradient<OnlyX>(
      u.data(), u.size3()[0], u.size3()[1], u.size3()[2], u.spacing()[0],
      u.spacing()[1], u.spacing()[2], u.halo_width(), order, callback);

  // Both methods invoke the same callback, and can be interleaved.
  grad.prepare();
  REQUIRE(callback_invoked == 1);
  grad.halo_preparation();
  REQUIRE(callback_invoked == 2);
  grad.prepare();
  REQUIRE(callback_invoked == 3);
}
