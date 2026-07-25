// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <openpfc/kernel/data/field_factory.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/data/domain.hpp>

TEST_CASE("field_from_subdomain creates Field with matching geometry",
          "[field_factory][unit]") {
  const int nx = 2, ny = 2, nz = 2;
  const pfc::Domain domain = pfc::domain::create({nx, ny, nz});
  const int num_ranks = 2;
  const pfc::Decomposition decomp =
      pfc::decomposition::create(domain, num_ranks);
  const int halo = 1;

  SECTION("All ranks get correctly sized fields") {
    for (int rank = 0; rank < num_ranks; ++rank) {
      pfc::data::Field<double, pfc::HostSpace> field =
          pfc::data::field_from_subdomain<double>(decomp, rank, halo);

      const pfc::Box3i expected_box = pfc::decomposition::local_box(decomp, rank);
      REQUIRE(field.box() == expected_box);
      REQUIRE(field.domain() == domain);
      REQUIRE(field.halo_width() == halo);
    }
  }

  SECTION("Zero halo width works correctly") {
    const int zero_halo = 0;
    for (int rank = 0; rank < num_ranks; ++rank) {
      pfc::data::Field<double, pfc::HostSpace> field =
          pfc::data::field_from_subdomain<double>(decomp, rank, zero_halo);

      const pfc::Box3i expected_box = pfc::decomposition::local_box(decomp, rank);
      REQUIRE(field.box() == expected_box);
      REQUIRE(field.halo_width() == zero_halo);
      REQUIRE(field.padded_extent(0) == expected_box.size[0]);
      REQUIRE(field.padded_extent(1) == expected_box.size[1]);
      REQUIRE(field.padded_extent(2) == expected_box.size[2]);
    }
  }

  SECTION("Different element types work") {
    pfc::data::Field<int, pfc::HostSpace> int_field =
        pfc::data::field_from_subdomain<int>(decomp, 0, halo);
    pfc::data::Field<std::complex<double>, pfc::HostSpace> complex_field =
        pfc::data::field_from_subdomain<std::complex<double>>(decomp, 1, halo);

    REQUIRE(int_field.halo_width() == halo);
    REQUIRE(complex_field.halo_width() == halo);
  }
}

TEST_CASE("field_from_subdomain throws on invalid rank", "[field_factory][unit]") {
  const pfc::Domain domain = pfc::domain::create({4, 4, 4});
  const pfc::Decomposition decomp = pfc::decomposition::create(domain, 2);

  SECTION("Negative rank throws") {
    REQUIRE_THROWS_AS(
        pfc::data::field_from_subdomain<double>(decomp, -1, 0),
        std::out_of_range);
  }

  SECTION("Rank >= num_domains throws") {
    REQUIRE_THROWS_AS(
        pfc::data::field_from_subdomain<double>(decomp, 2, 0),
        std::out_of_range);
    REQUIRE_THROWS_AS(
        pfc::data::field_from_subdomain<double>(decomp, 100, 0),
        std::out_of_range);
  }
}

TEST_CASE("field_from_subdomain with larger domain", "[field_factory][unit]") {
  const int nx = 8, ny = 4, nz = 2;
  const pfc::Domain domain = pfc::domain::create({nx, ny, nz});
  const int num_ranks = 4;
  const pfc::Decomposition decomp =
      pfc::decomposition::create(domain, num_ranks);

  SECTION("All ranks partition domain correctly") {
    int total_cells = 0;
    for (int rank = 0; rank < num_ranks; ++rank) {
      pfc::data::Field<double, pfc::HostSpace> field =
          pfc::data::field_from_subdomain<double>(decomp, rank, 0);

      const pfc::Box3i box = field.box();
      const int rank_cells = box.size[0] * box.size[1] * box.size[2];
      total_cells += rank_cells;

      REQUIRE(field.domain() == domain);
    }

    REQUIRE(total_cells == nx * ny * nz);
  }
}
