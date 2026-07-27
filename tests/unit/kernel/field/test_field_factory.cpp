// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Parity tests for pfc::data::field_from_subdomain: it must build a canonical
// Field whose layout reproduces the legacy LocalField::from_subdomain (halo 0)
// and PaddedBrick (halo n) construction bit-for-bit, so the M2 migration is a
// mechanical construction swap.

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/field/local_field.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>

using namespace pfc;

TEST_CASE("field_from_subdomain matches LocalField::from_subdomain (halo 0)",
          "[field_factory][unit]") {
  const int nx = 8, ny = 6, nz = 4;
  auto world = world::create(GridSize({nx, ny, nz}));
  auto decomp = decomposition::create(world, 1);
  auto lf = field::LocalField<double>::from_subdomain(decomp, /*rank=*/0, 0);

  auto f = data::field_from_subdomain<double>(decomp, /*rank=*/0, 0);

  REQUIRE(f.size() == lf.size());
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) REQUIRE(f.idx(i, j, k) == lf.idx(i, j, k));
}

TEST_CASE("field_from_subdomain matches PaddedBrick across the halo (halo n)",
          "[field_factory][unit]") {
  const int nx = 8, ny = 6, nz = 4, hw = 2;
  auto world = world::create(GridSize({nx, ny, nz}));
  auto decomp = decomposition::create(world, 1);
  field::PaddedBrick<double> pb(decomp, /*rank=*/0, hw);

  auto f = data::field_from_subdomain<double>(decomp, /*rank=*/0, hw);

  REQUIRE(f.size() == pb.size());
  for (int k = -hw; k < nz + hw; ++k)
    for (int j = -hw; j < ny + hw; ++j)
      for (int i = -hw; i < nx + hw; ++i) REQUIRE(f.idx(i, j, k) == pb.idx(i, j, k));
}

TEST_CASE("field_from_subdomain rejects a negative halo", "[field_factory][unit]") {
  auto world = world::create(GridSize({4, 4, 4}));
  auto decomp = decomposition::create(world, 1);
  REQUIRE_THROWS(data::field_from_subdomain<double>(decomp, /*rank=*/0, -1));
}
