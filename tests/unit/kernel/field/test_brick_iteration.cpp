// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <set>
#include <tuple>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/field/brick_iteration.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>

using namespace pfc;

namespace {

field::PaddedBrick<double> make_brick(int n, int hw) {
  auto world = world::create(GridSize({n, n, n}));
  auto decomp = decomposition::create(world, 1);
  return field::PaddedBrick<double>(decomp, /*rank=*/0, hw);
}

} // namespace

TEST_CASE("for_each_owned visits every owned cell exactly once",
          "[field][brick_iteration]") {
  auto u = make_brick(4, /*hw=*/1);
  std::set<std::tuple<int, int, int>> seen;
  field::for_each_owned(u, [&](int i, int j, int k) { seen.insert({i, j, k}); });
  REQUIRE(seen.size() == static_cast<std::size_t>(u.nx() * u.ny() * u.nz()));
  REQUIRE(*seen.begin() == std::tuple{0, 0, 0});
  REQUIRE(*seen.rbegin() == std::tuple{u.nx() - 1, u.ny() - 1, u.nz() - 1});
}

TEST_CASE("for_each yields Int3 in row-major order over every owned cell",
          "[field][brick_iteration]") {
  auto u = make_brick(4, /*hw=*/1);
  std::vector<pfc::Int3> seen;
  field::for_each(u, [&](const auto &idx) { seen.push_back(idx); });

  REQUIRE(seen.size() == static_cast<std::size_t>(u.nx() * u.ny() * u.nz()));
  REQUIRE(seen.front() == pfc::Int3{0, 0, 0});
  REQUIRE(seen.back() == pfc::Int3{u.nx() - 1, u.ny() - 1, u.nz() - 1});

  // Check the k-outer / j-middle / i-inner ordering: the i index must
  // monotonically advance until it wraps, then j, then k.
  for (std::size_t s = 1; s < seen.size(); ++s) {
    const auto &p = seen[s - 1];
    const auto &c = seen[s];
    const bool i_advance = (c[0] == p[0] + 1) && c[1] == p[1] && c[2] == p[2];
    const bool j_advance = (c[0] == 0) && c[1] == p[1] + 1 && c[2] == p[2];
    const bool k_advance = (c[0] == 0) && c[1] == 0 && c[2] == p[2] + 1;
    REQUIRE((i_advance || j_advance || k_advance));
  }

  // Confirm the body can write through brick(idx) without any (i, j, k)
  // unpacking — this is the workflow the heat3d_fd driver will use.
  field::for_each(u, [&](const auto &idx) { u(idx) = idx[0] + 10 * idx[1]; });
  for (int k = 0; k < u.nz(); ++k)
    for (int j = 0; j < u.ny(); ++j)
      for (int i = 0; i < u.nx(); ++i) REQUIRE(u(i, j, k) == i + 10 * j);
}

TEST_CASE("for_each_inner stays in [r, n-r) and obeys r=0 -> entire owned",
          "[field][brick_iteration]") {
  auto u = make_brick(5, /*hw=*/2);
  int count_r0 = 0, count_r1 = 0, count_r2 = 0;
  field::for_each_inner(u, 0, [&](int, int, int) { ++count_r0; });
  field::for_each_inner(u, 1, [&](int, int, int) { ++count_r1; });
  field::for_each_inner(u, 2, [&](int, int, int) { ++count_r2; });
  REQUIRE(count_r0 == u.nx() * u.ny() * u.nz());
  REQUIRE(count_r1 == 3 * 3 * 3);
  REQUIRE(count_r2 == 1);
}

TEST_CASE("for_each_inner is a no-op when n <= 2*r", "[field][brick_iteration]") {
  auto u = make_brick(4, /*hw=*/2);
  int count = 0;
  field::for_each_inner(u, 2, [&](int, int, int) { ++count; });
  REQUIRE(count == 0);
}

TEST_CASE("for_each_border covers owned-minus-inner exactly once",
          "[field][brick_iteration]") {
  for (int n : {4, 5, 6, 8}) {
    for (int r : {1, 2}) {
      if (n <= 2 * r) continue;
      auto u = make_brick(n, /*hw=*/r);
      std::set<std::tuple<int, int, int>> border;
      field::for_each_border(u, r, [&](int i, int j, int k) {
        const auto added = border.emplace(i, j, k).second;
        REQUIRE(added);
      });

      std::set<std::tuple<int, int, int>> inner;
      field::for_each_inner(u, r,
                            [&](int i, int j, int k) { inner.emplace(i, j, k); });

      REQUIRE(border.size() + inner.size() == static_cast<std::size_t>(n * n * n));

      for (const auto &c : border) REQUIRE(!inner.contains(c));

      for (const auto &b : border) {
        const int i = std::get<0>(b);
        const int j = std::get<1>(b);
        const int k = std::get<2>(b);
        const bool on_face = (i < r) || (i >= n - r) || (j < r) || (j >= n - r) ||
                             (k < r) || (k >= n - r);
        REQUIRE(on_face);
      }
    }
  }
}

TEST_CASE("for_each_border falls back to the whole owned region when n <= 2*r",
          "[field][brick_iteration]") {
  auto u = make_brick(4, /*hw=*/2);
  int count = 0;
  field::for_each_border(u, 2, [&](int, int, int) { ++count; });
  REQUIRE(count == u.nx() * u.ny() * u.nz());
}

TEST_CASE("stencil over inner region only reads owned cells (no halo dependency)",
          "[field][brick_iteration]") {
  auto u = make_brick(6, /*hw=*/1);
  for (int k = 0; k < u.nz(); ++k) {
    for (int j = 0; j < u.ny(); ++j) {
      for (int i = 0; i < u.nx(); ++i) u(i, j, k) = i + 10 * j + 100 * k;
    }
  }

  double accum = 0.0;
  field::for_each_inner(u, 1, [&](int i, int j, int k) {
    const double xx = u(i + 1, j, k) - 2 * u(i, j, k) + u(i - 1, j, k);
    const double yy = u(i, j + 1, k) - 2 * u(i, j, k) + u(i, j - 1, k);
    const double zz = u(i, j, k + 1) - 2 * u(i, j, k) + u(i, j, k - 1);
    accum += xx + yy + zz;
  });
  REQUIRE(accum == Catch::Approx(0.0));
}

TEST_CASE("for_each_owned_omp visits same set as serial counterpart",
          "[field][brick_iteration]") {
  auto u = make_brick(4, /*hw=*/0);
  std::vector<int> hits(u.nx() * u.ny() * u.nz(), 0);
  field::for_each_owned_omp(u, [&](int i, int j, int k) {
    const std::size_t lin = static_cast<std::size_t>(i) +
                            static_cast<std::size_t>(j) * u.nx() +
                            static_cast<std::size_t>(k) * u.nx() * u.ny();
    hits[lin] = 1;
  });
  for (int v : hits) REQUIRE(v == 1);
}
