// SPDX-License-Identifier: AGPL-3.0-or-later
//
// M2.1 parity tests for the canonical pfc::data::Field<T, MemorySpace>.
// The whole point of the merge is that ONE linearization reproduces the three
// legacy ones bit-for-bit, so these tests pin idx()/size()/coords() against
// real LocalField (halo 0) and PaddedBrick (halo n) instances built on the
// same single-rank geometry.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/field/local_field.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>

using namespace pfc;

namespace {
// Whole-domain owned box for a single-rank decomposition of GridSize{nx,ny,nz}.
Box3i whole_box(int nx, int ny, int nz) {
  return Box3i::from_bounds({0, 0, 0}, {nx - 1, ny - 1, nz - 1});
}
} // namespace

TEST_CASE("Field: idx matches LocalField bit-for-bit (halo 0)",
          "[grid_field][unit]") {
  const int nx = 8, ny = 6, nz = 4;
  auto world = pfc::domain::create_world({nx, ny, nz});
  auto decomp = decomposition::create(world, 1);
  auto lf = field::LocalField<double>::from_subdomain(decomp, /*rank=*/0, 0);

  data::Field<double> f(domain::create({nx, ny, nz}), whole_box(nx, ny, nz), 0);

  REQUIRE(f.size() == lf.size());
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) REQUIRE(f.idx(i, j, k) == lf.idx(i, j, k));
}

TEST_CASE("Field: idx matches PaddedBrick bit-for-bit across the halo (halo n)",
          "[grid_field][unit]") {
  const int nx = 8, ny = 6, nz = 4;
  const int hw = 2;
  auto world = pfc::domain::create_world({nx, ny, nz});
  auto decomp = decomposition::create(world, 1);
  field::PaddedBrick<double> pb(decomp, /*rank=*/0, hw);

  data::Field<double> f(domain::create({nx, ny, nz}), whole_box(nx, ny, nz), hw);

  REQUIRE(f.size() == pb.size());
  // Every addressable cell, including the halo slabs [-hw, n+hw).
  for (int k = -hw; k < nz + hw; ++k)
    for (int j = -hw; j < ny + hw; ++j)
      for (int i = -hw; i < nx + hw; ++i) REQUIRE(f.idx(i, j, k) == pb.idx(i, j, k));
}

TEST_CASE("Field: coordinate queries match LocalField", "[grid_field][unit]") {
  const int nx = 5, ny = 5, nz = 5;
  auto world = pfc::domain::create_world({nx, ny, nz});
  auto decomp = decomposition::create(world, 1);
  auto lf = field::LocalField<double>::from_subdomain(decomp, 0, 0);

  data::Field<double> f(domain::create({nx, ny, nz}), whole_box(nx, ny, nz), 0);

  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        REQUIRE(f.global(i, j, k) == lf.global(i, j, k));
        const auto fc = f.coords(i, j, k);
        const auto lc = lf.coords(i, j, k);
        REQUIRE(fc[0] == lc[0]);
        REQUIRE(fc[1] == lc[1]);
        REQUIRE(fc[2] == lc[2]);
      }
}

TEST_CASE("Field: padded storage size is prod(n + 2*halo)", "[grid_field][unit]") {
  data::Field<double> f(domain::create({4, 3, 2}),
                        Box3i::from_bounds({0, 0, 0}, {3, 2, 1}), 1);
  // (4+2)*(3+2)*(2+2) = 6*5*4 = 120
  REQUIRE(f.size() == 120u);
  REQUIRE(f.padded_extent(0) == 6);
  REQUIRE(f.padded_extent(1) == 5);
  REQUIRE(f.padded_extent(2) == 4);
  REQUIRE(f.local_size() == Int3{4, 3, 2});
  REQUIRE(f.halo_width() == 1);
}

TEST_CASE("Field: apply fills owned cells from physical coords",
          "[grid_field][unit]") {
  data::Field<double> f(domain::with_spacing({3, 1, 1}, {2.0, 1.0, 1.0}),
                        Box3i::from_bounds({0, 0, 0}, {2, 0, 0}), 0);
  f.apply([](double x, double, double) { return x; });
  // spacing 2, origin 0 -> owned cell x-coords are 0, 2, 4.
  REQUIRE(f(0, 0, 0) == Catch::Approx(0.0));
  REQUIRE(f(1, 0, 0) == Catch::Approx(2.0));
  REQUIRE(f(2, 0, 0) == Catch::Approx(4.0));
}

TEST_CASE("Field: rejects a negative halo or inconsistent box",
          "[grid_field][unit]") {
  REQUIRE_THROWS_AS(data::Field<double>(domain::create({2, 2, 2}),
                                        Box3i::from_bounds({0, 0, 0}, {1, 1, 1}),
                                        -1),
                    std::invalid_argument);
  // Hand-built box whose size disagrees with high-low+1.
  Box3i bad{};
  bad.low = {0, 0, 0};
  bad.high = {3, 3, 3};
  bad.size = {2, 2, 2};
  REQUIRE_THROWS_AS(data::Field<double>(domain::create({4, 4, 4}), bad, 0),
                    std::invalid_argument);
}

TEST_CASE("Field: a host-space field is one-sided and needs no transfer",
          "[grid_field][residency][unit]") {
  data::Field<double> f(domain::create({4, 4, 4}),
                        Box3i::from_bounds({0, 0, 0}, {3, 3, 3}), 0);
  REQUIRE_FALSE(f.residency().two_sided());
  REQUIRE(f.residency().host_valid());
  REQUIRE_FALSE(f.residency().host_needs_refresh());
  REQUIRE_FALSE(f.residency().device_needs_refresh());
}

TEST_CASE("Field: with_host_view brackets host access on a host-space field",
          "[grid_field][residency][unit]") {
  data::Field<double> f(domain::create({4, 4, 4}),
                        Box3i::from_bounds({0, 0, 0}, {3, 3, 3}), 0);
  bool called = false;
  f.with_host_view([&](double *data, std::size_t n) {
    called = true;
    REQUIRE(n == f.size());
    data[0] = 42.0; // write through the host buffer
  });
  REQUIRE(called);
  REQUIRE(f(0, 0, 0) == 42.0); // the write is visible through the field
  REQUIRE(f.residency().host_valid());
}
