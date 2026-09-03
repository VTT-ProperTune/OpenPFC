// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Complex-type and value-semantics tests for pfc::data::Field<T, MemorySpace>.
// Tests std::complex<double> element type with halo 0 and halo 2, and verifies
// that HostSpace Field copy construction produces independent buffers with
// identical geometry.

#include <catch2/catch_test_macros.hpp>
#include <complex>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>

using namespace pfc;

namespace {
// Whole-domain owned box for a single-rank decomposition of GridSize{nx,ny,nz}.
Box3i whole_box(int nx, int ny, int nz) {
  return Box3i::from_bounds({0, 0, 0}, {nx - 1, ny - 1, nz - 1});
}
} // namespace

// ============================================================================
// Complex element type tests
// ============================================================================

TEST_CASE("Field<std::complex<double>> size and indexing with halo 0",
          "[grid_field][complex][halo0]") {
  const int nx = 8, ny = 6, nz = 4;
  data::Field<std::complex<double>> f(domain::create({nx, ny, nz}),
                                      whole_box(nx, ny, nz), 0);

  SECTION("size equals padded volume") {
    const int halo = 0;
    const std::size_t expected_volume = static_cast<std::size_t>(nx + 2 * halo) *
                                         static_cast<std::size_t>(ny + 2 * halo) *
                                         static_cast<std::size_t>(nz + 2 * halo);
    REQUIRE(f.size() == expected_volume);
  }

  SECTION("idx and operator round-trip at owned cells") {
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          std::complex<double> value(1.0 + i, 2.0 + j);
          f(i, j, k) = value;
          REQUIRE(f.data()[f.idx(i, j, k)] == value);
        }
      }
    }
  }
}

TEST_CASE("Field<std::complex<double>> size and indexing with halo 2",
          "[grid_field][complex][halo2]") {
  const int nx = 8, ny = 6, nz = 4;
  const int halo = 2;
  data::Field<std::complex<double>> f(domain::create({nx, ny, nz}),
                                      whole_box(nx, ny, nz), halo);

  SECTION("size equals padded volume") {
    std::size_t expected_volume = static_cast<std::size_t>(nx + 2 * halo) *
                                   static_cast<std::size_t>(ny + 2 * halo) *
                                   static_cast<std::size_t>(nz + 2 * halo);
    REQUIRE(f.size() == expected_volume);
  }

  SECTION("idx and operator round-trip at owned cells") {
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          std::complex<double> value(i, j);
          f(i, j, k) = value;
          REQUIRE(f.data()[f.idx(i, j, k)] == value);
        }
      }
    }
  }

  SECTION("idx and operator round-trip at halo cells") {
    // Test a few representative halo cells
    std::array<std::array<int, 3>, 4> halo_indices = {
        {{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}, {-1, -1, -1}}};
    std::complex<double> halo_val(3.0, 4.0);

    for (const auto &idx : halo_indices) {
      f(idx[0], idx[1], idx[2]) = halo_val;
      REQUIRE(f.data()[f.idx(idx[0], idx[1], idx[2])] == halo_val);
    }
  }
}

// ============================================================================
// Value semantics tests
// ============================================================================

TEST_CASE("HostSpace Field copy has value semantics",
          "[grid_field][copy][semantics]") {
  const int nx = 6, ny = 6, nz = 6;
  const int halo = 1;
  data::Field<double, pfc::HostSpace> original(domain::create({nx, ny, nz}),
                                                whole_box(nx, ny, nz), halo);

  // Populate some values
  original(2, 2, 2) = 1.5;
  original(3, 3, 3) = 2.5;
  original(1, 1, 1) = 3.5;

  // Copy construct
  data::Field<double, pfc::HostSpace> copy = original;

  SECTION("copy has independent buffer") {
    // Modify copy
    copy(2, 2, 2) = 5.0;
    copy(3, 3, 3) = 6.0;
    copy(1, 1, 1) = 7.0;

    // Original should be unchanged
    REQUIRE(original(2, 2, 2) == 1.5);
    REQUIRE(original(3, 3, 3) == 2.5);
    REQUIRE(original(1, 1, 1) == 3.5);

    // Copy should have new values
    REQUIRE(copy(2, 2, 2) == 5.0);
    REQUIRE(copy(3, 3, 3) == 6.0);
    REQUIRE(copy(1, 1, 1) == 7.0);
  }

  SECTION("copy has identical box") {
    const auto &orig_box = original.box();
    const auto &copy_box = copy.box();

    REQUIRE(orig_box.low[0] == copy_box.low[0]);
    REQUIRE(orig_box.low[1] == copy_box.low[1]);
    REQUIRE(orig_box.low[2] == copy_box.low[2]);

    REQUIRE(orig_box.high[0] == copy_box.high[0]);
    REQUIRE(orig_box.high[1] == copy_box.high[1]);
    REQUIRE(orig_box.high[2] == copy_box.high[2]);

    REQUIRE(orig_box.size[0] == copy_box.size[0]);
    REQUIRE(orig_box.size[1] == copy_box.size[1]);
    REQUIRE(orig_box.size[2] == copy_box.size[2]);
  }

  SECTION("copy has identical halo_width") {
    REQUIRE(copy.halo_width() == original.halo_width());
  }

  SECTION("copy has identical domain") {
    const auto &orig_domain = original.domain();
    const auto &copy_domain = copy.domain();

    // Check spacing
    const auto &orig_spacing = pfc::domain::get_spacing(orig_domain);
    const auto &copy_spacing = pfc::domain::get_spacing(copy_domain);

    REQUIRE(orig_spacing[0] == copy_spacing[0]);
    REQUIRE(orig_spacing[1] == copy_spacing[1]);
    REQUIRE(orig_spacing[2] == copy_spacing[2]);

    // Check origin
    const auto &orig_origin = pfc::domain::get_origin(orig_domain);
    const auto &copy_origin = pfc::domain::get_origin(copy_domain);

    REQUIRE(orig_origin[0] == copy_origin[0]);
    REQUIRE(orig_origin[1] == copy_origin[1]);
    REQUIRE(orig_origin[2] == copy_origin[2]);
  }

  SECTION("copy has same padded extents") {
    REQUIRE(copy.padded_extent(0) == original.padded_extent(0));
    REQUIRE(copy.padded_extent(1) == original.padded_extent(1));
    REQUIRE(copy.padded_extent(2) == original.padded_extent(2));
  }

  SECTION("copy has same total size") {
    REQUIRE(copy.size() == original.size());
  }
}
