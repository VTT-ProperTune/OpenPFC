// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>

using namespace pfc;
using Catch::Approx;

TEST_CASE("PaddedBrick: storage size matches (n+2hw)^3 and idx round-trip",
          "[field][padded_brick]") {
  auto world = world::create(GridSize({8, 6, 4}));
  auto decomp = decomposition::create(world, 1);

  const int hw = 2;
  field::PaddedBrick<double> u(decomp, /*rank=*/0, hw);

  REQUIRE(u.nx() == 8);
  REQUIRE(u.ny() == 6);
  REQUIRE(u.nz() == 4);
  REQUIRE(u.nx_padded() == 8 + 2 * hw);
  REQUIRE(u.ny_padded() == 6 + 2 * hw);
  REQUIRE(u.nz_padded() == 4 + 2 * hw);
  REQUIRE(u.halo_width() == hw);

  const std::size_t expected_size = static_cast<std::size_t>(u.nx_padded()) *
                                    static_cast<std::size_t>(u.ny_padded()) *
                                    static_cast<std::size_t>(u.nz_padded());
  REQUIRE(u.size() == expected_size);
  REQUIRE(u.vec().size() == expected_size);

  REQUIRE(u.idx(-hw, -hw, -hw) == 0);
  REQUIRE(u.idx(u.nx() + hw - 1, u.ny() + hw - 1, u.nz() + hw - 1) ==
          expected_size - 1);
  REQUIRE(u.idx(0, 0, 0) == static_cast<std::size_t>(hw) +
                                static_cast<std::size_t>(hw) *
                                    static_cast<std::size_t>(u.nx_padded()) +
                                static_cast<std::size_t>(hw) *
                                    static_cast<std::size_t>(u.nx_padded()) *
                                    static_cast<std::size_t>(u.ny_padded()));
}

TEST_CASE("PaddedBrick: zero-initialized on construction", "[field][padded_brick]") {
  auto world = world::create(GridSize({4, 4, 4}));
  auto decomp = decomposition::create(world, 1);

  field::PaddedBrick<double> u(decomp, 0, /*hw=*/1);
  bool values_are_zero = true;
  for (double v : u.vec()) {
    values_are_zero &= v == 0.0;
  }
  REQUIRE(values_are_zero);
}

TEST_CASE("PaddedBrick: operator() reaches halo cells in [-hw, n+hw)",
          "[field][padded_brick]") {
  auto world = world::create(GridSize({4, 4, 4}));
  auto decomp = decomposition::create(world, 1);

  const int hw = 1;
  field::PaddedBrick<int> u(decomp, 0, hw);

  bool values_match = true;
  for (int k = -hw; k < u.nz() + hw; ++k) {
    for (int j = -hw; j < u.ny() + hw; ++j) {
      for (int i = -hw; i < u.nx() + hw; ++i) {
        u(i, j, k) = 100 * (k + hw) + 10 * (j + hw) + (i + hw);
      }
    }
  }

  for (int k = -hw; k < u.nz() + hw; ++k) {
    for (int j = -hw; j < u.ny() + hw; ++j) {
      for (int i = -hw; i < u.nx() + hw; ++i) {
        values_match &= u(i, j, k) == 100 * (k + hw) + 10 * (j + hw) + (i + hw);
      }
    }
  }
  REQUIRE(values_match);
}

TEST_CASE("PaddedBrick: apply fills only owned cells, halos stay zero",
          "[field][padded_brick]") {
  auto world = world::create(GridSize({4, 4, 4}));
  auto decomp = decomposition::create(world, 1);

  const int hw = 1;
  field::PaddedBrick<double> u(decomp, 0, hw);

  u.apply([](double x, double y, double z) { return x + 10 * y + 100 * z; });

  bool values_match = true;
  for (int k = 0; k < u.nz(); ++k) {
    for (int j = 0; j < u.ny(); ++j) {
      for (int i = 0; i < u.nx(); ++i) {
        const auto p = u.global_coords(i, j, k);
        values_match &= u(i, j, k) == Approx(p[0] + 10 * p[1] + 100 * p[2]);
      }
    }
  }

  for (int i = -hw; i < u.nx() + hw; ++i) {
    values_match &= u(i, -1, 0) == 0.0 && u(i, u.ny(), 0) == 0.0;
  }
  for (int j = -hw; j < u.ny() + hw; ++j) {
    values_match &= u(-1, j, 0) == 0.0 && u(u.nx(), j, 0) == 0.0;
  }
  for (int j = 0; j < u.ny(); ++j) {
    for (int i = 0; i < u.nx(); ++i) {
      values_match &= u(i, j, -1) == 0.0 && u(i, j, u.nz()) == 0.0;
    }
  }
  REQUIRE(values_match);
}

TEST_CASE("PaddedBrick: global_coords extrapolates across the halo ring",
          "[field][padded_brick]") {
  auto world = world::create(GridSize({4, 4, 4}));
  auto decomp = decomposition::create(world, 1);

  const int hw = 2;
  field::PaddedBrick<double> u(decomp, 0, hw);

  const auto p0 = u.global_coords(0, 0, 0);
  const auto pneg = u.global_coords(-1, 0, 0);
  const auto ppos = u.global_coords(u.nx(), 0, 0);

  const double dx = u.spacing()[0];
  REQUIRE(pneg[0] == Approx(p0[0] - dx));
  REQUIRE(ppos[0] == Approx(p0[0] + u.nx() * dx));
}

TEST_CASE("PaddedBrick: hw=0 reduces to plain owned-only buffer",
          "[field][padded_brick]") {
  auto world = world::create(GridSize({3, 3, 3}));
  auto decomp = decomposition::create(world, 1);

  field::PaddedBrick<double> u(decomp, 0, /*hw=*/0);
  REQUIRE(u.nx_padded() == u.nx());
  REQUIRE(u.size() == 27);
  REQUIRE(u.idx(0, 0, 0) == 0);
  REQUIRE(u.idx(2, 2, 2) == 26);
}

TEST_CASE("PaddedBrick: rejects negative halo width", "[field][padded_brick]") {
  auto world = world::create(GridSize({4, 4, 4}));
  auto decomp = decomposition::create(world, 1);
  REQUIRE_THROWS_AS(field::PaddedBrick<double>(decomp, 0, -1),
                    std::invalid_argument);
}

TEST_CASE("PaddedBrick: carries decomposition and rank along with hw",
          "[field][padded_brick]") {
  auto world = world::create(GridSize({8, 6, 4}));
  auto decomp = decomposition::create(world, 1);

  const int hw = 2;
  field::PaddedBrick<double> u(decomp, /*rank=*/0, hw);

  REQUIRE(u.rank() == 0);
  REQUIRE(u.halo_width() == hw);

  // Decomposition is stored by value: PaddedBrick stays valid even after the
  // factory's local Decomposition object goes out of scope (mirrors the
  // lifetime guarantee in test_decomposition_lifetime.cpp).
  const auto &owned_decomp = u.decomposition();
  REQUIRE(world::get_size(decomposition::get_subworld(owned_decomp, 0)) ==
          world::get_size(decomposition::get_subworld(decomp, 0)));
}

TEST_CASE("PaddedBrick: indices() walks every owned cell in row-major order",
          "[field][padded_brick]") {
  auto world = world::create(GridSize({4, 3, 2}));
  auto decomp = decomposition::create(world, 1);

  field::PaddedBrick<int> u(decomp, /*rank=*/0, /*hw=*/1);

  std::size_t count = 0;
  pfc::Int3 last{-1, -1, -1};
  bool indices_are_owned = true;
  for (const auto idx : u.indices()) {
    indices_are_owned &= idx[0] >= 0 && idx[0] < u.nx() && idx[1] >= 0 &&
                         idx[1] < u.ny() && idx[2] >= 0 && idx[2] < u.nz();
    last = idx;
    ++count;
  }
  REQUIRE(indices_are_owned);
  REQUIRE(count == static_cast<std::size_t>(u.nx() * u.ny() * u.nz()));
  REQUIRE(last == pfc::Int3{u.nx() - 1, u.ny() - 1, u.nz() - 1});
}

TEST_CASE("PaddedBrick: indices_inner(r) skips r-thick boundary slab",
          "[field][padded_brick]") {
  auto world = world::create(GridSize({6, 6, 6}));
  auto decomp = decomposition::create(world, 1);

  field::PaddedBrick<int> u(decomp, /*rank=*/0, /*hw=*/2);

  const int r = 2;
  std::size_t count = 0;
  bool indices_are_inner = true;
  for (const auto idx : u.indices_inner(r)) {
    indices_are_inner &= idx[0] >= r && idx[0] < u.nx() - r && idx[1] >= r &&
                         idx[1] < u.ny() - r && idx[2] >= r && idx[2] < u.nz() - r;
    ++count;
  }
  REQUIRE(indices_are_inner);
  REQUIRE(count == static_cast<std::size_t>((u.nx() - 2 * r) * (u.ny() - 2 * r) *
                                            (u.nz() - 2 * r)));

  // Empty range when r is too large for the owned core.
  std::size_t empty_count = 0;
  for (const auto idx : u.indices_inner(/*r=*/u.nx())) {
    (void)idx;
    ++empty_count;
  }
  REQUIRE(empty_count == 0);
}

TEST_CASE("PaddedBrick: Int3 overloads of idx/operator() match scalar form",
          "[field][padded_brick]") {
  auto world = world::create(GridSize({3, 3, 3}));
  auto decomp = decomposition::create(world, 1);

  field::PaddedBrick<int> u(decomp, /*rank=*/0, /*hw=*/1);

  for (int k = 0; k < u.nz(); ++k) {
    for (int j = 0; j < u.ny(); ++j) {
      for (int i = 0; i < u.nx(); ++i) {
        u(pfc::Int3{i, j, k}) = i + 10 * j + 100 * k;
      }
    }
  }
  bool overloads_match = true;
  for (const auto idx : u.indices())
    overloads_match &= u(idx) == idx[0] + 10 * idx[1] + 100 * idx[2] &&
                       u.idx(idx) == u.idx(idx[0], idx[1], idx[2]);
  REQUIRE(overloads_match);
}

TEST_CASE("PaddedBrick: throws on halo width overflow in extent calculation",
          "[field][padded_brick]") {
  auto world = world::create(GridSize({1000000, 1, 1}));
  auto decomp = decomposition::create(world, 1);

  // Large halo width that causes nx + 2*hw to exceed INT_MAX
  // INT_MAX on most systems is 2147483647
  // 1000000 + 2*1500000000 = 1000000 + 3000000000 = 4000000000 > INT_MAX
  REQUIRE_THROWS_AS(
      field::PaddedBrick<double>(decomp, /*rank=*/0, /*hw=*/1500000000),
      std::overflow_error);
}

TEST_CASE("PaddedBrick: throws on product overflow in storage allocation",
          "[field][padded_brick]") {
  auto world = world::create(GridSize({1000000, 2000000, 1}));
  auto decomp = decomposition::create(world, 1);

  // Large extents that cause nx*ny*nz to overflow size_t
  // On binary64 systems, size_t is 64-bit, but the product of three int32 values
  // can still exceed the max before the product check
  // Use a smaller halo width to avoid extent overflow but still hit product overflow
  REQUIRE_THROWS_AS(
      field::PaddedBrick<double>(decomp, /*rank=*/0, /*hw=*/1000),
      std::overflow_error);
}

TEST_CASE("PaddedBrick: throws on negative halo width", "[field][padded_brick]") {
  auto world = world::create(GridSize({4, 4, 4}));
  auto decomp = decomposition::create(world, 1);
  REQUIRE_THROWS_AS(field::PaddedBrick<double>(decomp, 0, -1),
                    std::invalid_argument);
}
