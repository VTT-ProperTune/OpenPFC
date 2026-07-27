// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/field/local_field.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>

using namespace pfc;
using Catch::Approx;

namespace {

// Helper template to verify bit-for-bit parity between field_from_subdomain
// and LocalField::from_subdomain for halo=0
template <typename T>
void verify_field_parity_with_local_field(const decomposition::Decomposition& decomp,
                                           int rank) {
  auto new_field = pfc::data::field_from_subdomain<T>(decomp, rank, 0);
  auto legacy_field = field::LocalField<T>::from_subdomain(decomp, rank, 0);

  // Compare sizes
  REQUIRE(new_field.size() == legacy_field.size());
  REQUIRE(new_field.local_size() == legacy_field.size3());

  // Compare indexing across all owned cells
  const auto size = new_field.local_size();
  for (int k = 0; k < size[2]; ++k) {
    for (int j = 0; j < size[1]; ++j) {
      for (int i = 0; i < size[0]; ++i) {
        REQUIRE(new_field.idx(i, j, k) == legacy_field.idx(i, j, k));
      }
    }
  }

  // Compare coordinate queries
  for (int k = 0; k < size[2]; ++k) {
    for (int j = 0; j < size[1]; ++j) {
      for (int i = 0; i < size[0]; ++i) {
        REQUIRE(new_field.global(i, j, k) == legacy_field.global(i, j, k));
        const auto new_coords = new_field.coords(i, j, k);
        const auto legacy_coords = legacy_field.coords(i, j, k);
        REQUIRE(new_coords[0] == Approx(legacy_coords[0]));
        REQUIRE(new_coords[1] == Approx(legacy_coords[1]));
        REQUIRE(new_coords[2] == Approx(legacy_coords[2]));
      }
    }
  }

  // Verify data access works identically
  for (int k = 0; k < size[2]; ++k) {
    for (int j = 0; j < size[1]; ++j) {
      for (int i = 0; i < size[0]; ++i) {
        const T test_value = static_cast<T>(i + size[0] * (j + size[1] * k));
        new_field(i, j, k) = test_value;
        legacy_field(i, j, k) = test_value;
        REQUIRE(new_field(i, j, k) == legacy_field(i, j, k));
      }
    }
  }
}

// Helper template to verify bit-for-bit parity between field_from_subdomain
// and PaddedBrick for halo=n
template <typename T>
void verify_field_parity_with_padded_brick(const decomposition::Decomposition& decomp,
                                            int rank, int halo) {
  auto new_field = pfc::data::field_from_subdomain<T>(decomp, rank, halo);
  auto padded_field = field::PaddedBrick<T>(decomp, rank, halo);

  // Compare sizes (includes halo padding)
  REQUIRE(new_field.size() == padded_field.size());

  // Compare indexing across all addressable cells including halo
  const auto size = new_field.local_size();
  for (int k = -halo; k < size[2] + halo; ++k) {
    for (int j = -halo; j < size[1] + halo; ++j) {
      for (int i = -halo; i < size[0] + halo; ++i) {
        REQUIRE(new_field.idx(i, j, k) == padded_field.idx(i, j, k));
      }
    }
  }

  // Compare coordinate queries for owned cells
  for (int k = 0; k < size[2]; ++k) {
    for (int j = 0; j < size[1]; ++j) {
      for (int i = 0; i < size[0]; ++i) {
        REQUIRE(new_field.global(i, j, k) == padded_field.global(i, j, k));
        const auto new_coords = new_field.coords(i, j, k);
        const auto padded_coords = padded_field.global_coords(i, j, k);
        REQUIRE(new_coords[0] == Approx(padded_coords[0]));
        REQUIRE(new_coords[1] == Approx(padded_coords[1]));
        REQUIRE(new_coords[2] == Approx(padded_coords[2]));
      }
    }
  }

  // Verify data access works identically across full padded range
  for (int k = -halo; k < size[2] + halo; ++k) {
    for (int j = -halo; j < size[1] + halo; ++j) {
      for (int i = -halo; i < size[0] + halo; ++i) {
        const T test_value = static_cast<T>(i + size[0] * (j + size[1] * k));
        new_field(i, j, k) = test_value;
        padded_field(i, j, k) = test_value;
        REQUIRE(new_field(i, j, k) == padded_field(i, j, k));
      }
    }
  }
}

} // namespace

TEST_CASE("field_from_subdomain: parity with LocalField (halo=0)", "[field_factory][unit]") {
  const int nx = 8, ny = 6, nz = 4;
  auto world = world::create(GridSize({nx, ny, nz}));
  auto decomp = decomposition::create(world, 1);

  verify_field_parity_with_local_field<double>(decomp, 0);
  verify_field_parity_with_local_field<float>(decomp, 0);
}

TEST_CASE("field_from_subdomain: parity with LocalField for multiple ranks",
          "[field_factory][unit]") {
  const int nx = 12, ny = 8, nz = 6;
  auto world = world::create(GridSize({nx, ny, nz}));
  auto decomp = decomposition::create(world, 4); // 4 ranks

  for (int rank = 0; rank < 4; ++rank) {
    verify_field_parity_with_local_field<double>(decomp, rank);
  }
}

TEST_CASE("field_from_subdomain: parity with PaddedBrick (halo=n)", "[field_factory][unit]") {
  const int nx = 8, ny = 6, nz = 4;
  auto world = world::create(GridSize({nx, ny, nz}));
  auto decomp = decomposition::create(world, 1);

  for (int halo : {0, 1, 2, 3}) {
    verify_field_parity_with_padded_brick<double>(decomp, 0, halo);
    verify_field_parity_with_padded_brick<float>(decomp, 0, halo);
  }
}

TEST_CASE("field_from_subdomain: parity with PaddedBrick for multiple ranks",
          "[field_factory][unit]") {
  const int nx = 12, ny = 8, nz = 6;
  auto world = world::create(GridSize({nx, ny, nz}));
  auto decomp = decomposition::create(world, 8); // 8 ranks

  for (int rank = 0; rank < 8; ++rank) {
    verify_field_parity_with_padded_brick<double>(decomp, rank, 1);
    verify_field_parity_with_padded_brick<float>(decomp, rank, 2);
  }
}

TEST_CASE("field_from_subdomain: rejects negative halo", "[field_factory][unit]") {
  const int nx = 8, ny = 6, nz = 4;
  auto world = world::create(GridSize({nx, ny, nz}));
  auto decomp = decomposition::create(world, 1);

  REQUIRE_THROWS_AS(pfc::data::field_from_subdomain<double>(decomp, 0, -1),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::data::field_from_subdomain<double>(decomp, 0, -10),
                    std::invalid_argument);
}

TEST_CASE("field_from_subdomain: geometry matches decomposition", "[field_factory][unit]") {
  const int nx = 16, ny = 12, nz = 8;
  auto domain = domain::with_spacing({nx, ny, nz}, {1.5, 2.0, 2.5});
  auto decomp = decomposition::create(domain, 4);

  for (int rank = 0; rank < 4; ++rank) {
    auto field = pfc::data::field_from_subdomain<double>(decomp, rank, 2);

    // Verify domain geometry is preserved
    const auto& field_domain = field.domain();
    const auto& decomp_domain = decomposition::domain(decomp);
    REQUIRE(pfc::domain::get_origin(field_domain) == pfc::domain::get_origin(decomp_domain));
    REQUIRE(pfc::domain::get_spacing(field_domain) == pfc::domain::get_spacing(decomp_domain));
    REQUIRE(pfc::domain::get_size(field_domain) == pfc::domain::get_size(decomp_domain));

    // Verify local box matches decomposition
    const auto& field_box = field.box();
    const auto decomp_box = decomposition::local_box(decomp, rank);
    REQUIRE(field_box.low == decomp_box.low);
    REQUIRE(field_box.high == decomp_box.high);
    REQUIRE(field_box.size == decomp_box.size);

    // Verify halo width matches
    REQUIRE(field.halo_width() == 2);
  }
}

TEST_CASE("field_from_subdomain: creates valid field with varying sizes",
          "[field_factory][unit]") {
  auto domain = domain::create({4, 4, 4});
  auto decomp = decomposition::create(domain, 1);

  for (int halo = 0; halo <= 2; ++halo) {
    auto field = pfc::data::field_from_subdomain<double>(decomp, 0, halo);

    // Verify field is properly constructed
    REQUIRE(field.size() > 0);
    REQUIRE(field.local_size() == Int3{4, 4, 4});
    REQUIRE(field.halo_width() == halo);

    // Verify we can write and read values
    field.apply([](double x, double y, double z) { return x + y + z; });
    REQUIRE(field(0, 0, 0) == Approx(0.0)); // origin
  }
}
