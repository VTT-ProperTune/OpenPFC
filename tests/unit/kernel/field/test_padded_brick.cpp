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
  for (double v : u.vec()) {
    REQUIRE(v == 0.0);
  }
}

TEST_CASE("PaddedBrick: operator() reaches halo cells in [-hw, n+hw)",
          "[field][padded_brick]") {
  auto world = world::create(GridSize({4, 4, 4}));
  auto decomp = decomposition::create(world, 1);

  const int hw = 1;
  field::PaddedBrick<int> u(decomp, 0, hw);

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
        REQUIRE(u(i, j, k) == 100 * (k + hw) + 10 * (j + hw) + (i + hw));
      }
    }
  }
}

TEST_CASE("PaddedBrick: apply fills only owned cells, halos stay zero",
          "[field][padded_brick]") {
  auto world = world::create(GridSize({4, 4, 4}));
  auto decomp = decomposition::create(world, 1);

  const int hw = 1;
  field::PaddedBrick<double> u(decomp, 0, hw);

  u.apply([](double x, double y, double z) { return x + 10 * y + 100 * z; });

  for (int k = 0; k < u.nz(); ++k) {
    for (int j = 0; j < u.ny(); ++j) {
      for (int i = 0; i < u.nx(); ++i) {
        const auto p = u.global_coords(i, j, k);
        REQUIRE(u(i, j, k) == Approx(p[0] + 10 * p[1] + 100 * p[2]));
      }
    }
  }

  for (int i = -hw; i < u.nx() + hw; ++i) {
    REQUIRE(u(i, -1, 0) == 0.0);
    REQUIRE(u(i, u.ny(), 0) == 0.0);
  }
  for (int j = -hw; j < u.ny() + hw; ++j) {
    REQUIRE(u(-1, j, 0) == 0.0);
    REQUIRE(u(u.nx(), j, 0) == 0.0);
  }
  for (int j = 0; j < u.ny(); ++j) {
    for (int i = 0; i < u.nx(); ++i) {
      REQUIRE(u(i, j, -1) == 0.0);
      REQUIRE(u(i, j, u.nz()) == 0.0);
    }
  }
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
