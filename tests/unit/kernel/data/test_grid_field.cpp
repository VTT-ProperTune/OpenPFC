// SPDX-License-Identifier: AGPL-3.0-or-later
//
// M2.1 correctness tests for the canonical pfc::data::Field<T, MemorySpace>.
// These tests verify idx()/size()/coords() expectations against closed-form
// mathematical derivations rather than legacy container instances, proving
// the field layout matches the expected row-major (x-fastest) linearization
// for both padded and unpadded cases.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

using namespace pfc;

namespace {
// Whole-domain owned box for a single-rank decomposition of GridSize{nx,ny,nz}.
Box3i whole_box(int nx, int ny, int nz) {
  return Box3i::from_bounds({0, 0, 0}, {nx - 1, ny - 1, nz - 1});
}
} // namespace

TEST_CASE("Field: idx and size for unpadded field (halo 0)",
          "[grid_field][unit]") {
  const int nx = 8, ny = 6, nz = 4;
  data::Field<double> f(domain::create({nx, ny, nz}), whole_box(nx, ny, nz), 0);

  // Expected size for unpadded field (halo 0)
  REQUIRE(f.size() == static_cast<std::size_t>(nx * ny * nz));

  // For row-major (x-fastest) layout: idx = i + j*nx + k*nx*ny
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        std::size_t expected_idx =
            static_cast<std::size_t>(i) +
            static_cast<std::size_t>(j) * nx +
            static_cast<std::size_t>(k) * nx * ny;
        REQUIRE(f.idx(i, j, k) == expected_idx);
      }
}

TEST_CASE("Field: idx and size across padded halo (halo n)",
          "[grid_field][unit]") {
  const int nx = 8, ny = 6, nz = 4;
  const int hw = 2;
  data::Field<double> f(domain::create({nx, ny, nz}), whole_box(nx, ny, nz), hw);

  // Expected size for padded field: prod(size + 2*halo)
  REQUIRE(f.size() == static_cast<std::size_t>((nx + 2 * hw) * (ny + 2 * hw) * (nz + 2 * hw)));

  // For row-major (x-fastest) padded layout:
  // idx = (i+hw) + (j+hw)*(nx+2*hw) + (k+hw)*(nx+2*hw)*(ny+2*hw)
  // Test every addressable cell, including the halo slabs [-hw, n+hw).
  const int npx = nx + 2 * hw;
  const int npy = ny + 2 * hw;
  const int npz = nz + 2 * hw;
  for (int k = -hw; k < nz + hw; ++k)
    for (int j = -hw; j < ny + hw; ++j)
      for (int i = -hw; i < nx + hw; ++i) {
        std::size_t expected_idx =
            (static_cast<std::size_t>(i) + hw) +
            (static_cast<std::size_t>(j) + hw) * npx +
            (static_cast<std::size_t>(k) + hw) * npx * npy;
        REQUIRE(f.idx(i, j, k) == expected_idx);
      }
}

TEST_CASE("Field: coordinate queries (global and physical)", "[grid_field][unit]") {
  const int nx = 5, ny = 5, nz = 5;
  data::Field<double> f(domain::create({nx, ny, nz}), whole_box(nx, ny, nz), 0);

  // For a single-rank decomposition starting at origin:
  // - global(i,j,k) == {i,j,k} (local box starts at {0,0,0})
  // - coords(i,j,k) == origin + spacing * {i,j,k} (defaults to 0,0,0 + 1.0,1.0,1.0)
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        // Global indices: local + low offset (which is 0 for whole_box starting at 0)
        const auto g = f.global(i, j, k);
        REQUIRE(g[0] == i);
        REQUIRE(g[1] == j);
        REQUIRE(g[2] == k);

        // Physical coordinates: origin + spacing * local
        const auto c = f.coords(i, j, k);
        REQUIRE(c[0] == Catch::Approx(static_cast<double>(i)));
        REQUIRE(c[1] == Catch::Approx(static_cast<double>(j)));
        REQUIRE(c[2] == Catch::Approx(static_cast<double>(k)));
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
