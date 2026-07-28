// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <complex>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>

using namespace Catch::Matchers;
using namespace pfc;

TEST_CASE("Field geometry and size queries", "[field][geometry]") {
  const int nx = 10, ny = 15, nz = 20;
  const pfc::Domain domain = pfc::domain::create({nx, ny, nz});
  const int num_ranks = 1;
  const pfc::Decomposition decomp = pfc::decomposition::create(domain, num_ranks);
  const int rank = 0;
  const int halo = 0;

  SECTION("Field size matches decomposition") {
    pfc::data::Field<double, pfc::HostSpace> field =
        pfc::data::field_from_subdomain<double>(decomp, rank, halo);

    REQUIRE(field.local_size()[0] == nx);
    REQUIRE(field.local_size()[1] == ny);
    REQUIRE(field.local_size()[2] == nz);
    REQUIRE(field.size3()[0] == nx);
    REQUIRE(field.size3()[1] == ny);
    REQUIRE(field.size3()[2] == nz);
    REQUIRE(field.global_size()[0] == nx);
    REQUIRE(field.global_size()[1] == ny);
    REQUIRE(field.global_size()[2] == nz);
  }

  SECTION("Field domain and spacing are correct") {
    pfc::data::Field<double, pfc::HostSpace> field =
        pfc::data::field_from_subdomain<double>(decomp, rank, halo);

    const pfc::Domain &field_domain = field.domain();
    REQUIRE(pfc::domain::get_size(field_domain)[0] == nx);
    REQUIRE(pfc::domain::get_size(field_domain)[1] == ny);
    REQUIRE(pfc::domain::get_size(field_domain)[2] == nz);

    const pfc::Real3 &spacing = field.spacing();
    REQUIRE(spacing[0] == 1.0);
    REQUIRE(spacing[1] == 1.0);
    REQUIRE(spacing[2] == 1.0);

    const pfc::Real3 &origin = field.origin();
    REQUIRE(origin[0] == 0.0);
    REQUIRE(origin[1] == 0.0);
    REQUIRE(origin[2] == 0.0);
  }

  SECTION("Field box and halo are correct") {
    const int test_halo = 2;
    pfc::data::Field<double, pfc::HostSpace> field =
        pfc::data::field_from_subdomain<double>(decomp, rank, test_halo);

    REQUIRE(field.halo_width() == test_halo);
    REQUIRE(field.storage_halo() == test_halo);

    const pfc::Box3i &box = field.box();
    REQUIRE(box.size[0] == nx);
    REQUIRE(box.size[1] == ny);
    REQUIRE(box.size[2] == nz);
  }

  SECTION("Custom domain with non-unit spacing and origin") {
    const pfc::Real3 custom_spacing{2.0, 3.0, 4.0};
    const pfc::Real3 custom_origin{1.0, 2.0, 3.0};
    const pfc::Domain custom_domain =
        pfc::domain::create(pfc::GridSize({nx, ny, nz}),
                            pfc::PhysicalOrigin(custom_origin),
                            pfc::GridSpacing(custom_spacing));
    const pfc::Decomposition custom_decomp =
        pfc::decomposition::create(custom_domain, num_ranks);

    pfc::data::Field<double, pfc::HostSpace> field =
        pfc::data::field_from_subdomain<double>(custom_decomp, rank, halo);

    const pfc::Real3 &spacing = field.spacing();
    REQUIRE(spacing[0] == custom_spacing[0]);
    REQUIRE(spacing[1] == custom_spacing[1]);
    REQUIRE(spacing[2] == custom_spacing[2]);

    const pfc::Real3 &origin = field.origin();
    REQUIRE(origin[0] == custom_origin[0]);
    REQUIRE(origin[1] == custom_origin[1]);
    REQUIRE(origin[2] == custom_origin[2]);
  }
}

TEST_CASE("Field apply function", "[field][apply]") {
  const int nx = 5, ny = 5, nz = 5;
  const pfc::Domain domain = pfc::domain::create({nx, ny, nz});
  const int num_ranks = 1;
  const pfc::Decomposition decomp = pfc::decomposition::create(domain, num_ranks);
  const int rank = 0;
  const int halo = 0;

  SECTION("Apply with coordinate function (double, double, double)") {
    pfc::data::Field<double, pfc::HostSpace> field =
        pfc::data::field_from_subdomain<double>(decomp, rank, halo);

    field.apply([](double x, double y, double z) {
      return x + y + z;
    });

    // Check specific indices
    REQUIRE(field(0, 0, 0) == 0.0);
    REQUIRE(field(1, 0, 0) == 1.0);
    REQUIRE(field(0, 1, 0) == 1.0);
    REQUIRE(field(0, 0, 1) == 1.0);
    REQUIRE(field(1, 1, 1) == 3.0);
    REQUIRE(field(2, 3, 4) == 9.0);
  }

  SECTION("Apply with Real3 coordinate function") {
    pfc::data::Field<double, pfc::HostSpace> field =
        pfc::data::field_from_subdomain<double>(decomp, rank, halo);

    field.apply([](const pfc::Real3 &coords) {
      return coords[0] * 10.0 + coords[1] * 1.0 + coords[2] * 0.1;
    });

    // Check specific indices
    REQUIRE_THAT(field(0, 0, 0), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(field(1, 0, 0), WithinAbs(10.0, 1e-10));
    REQUIRE_THAT(field(0, 1, 0), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(field(0, 0, 1), WithinAbs(0.1, 1e-10));
    REQUIRE_THAT(field(1, 1, 1), WithinAbs(11.1, 1e-10));
  }

  SECTION("Apply with complex function") {
    pfc::data::Field<std::complex<double>, pfc::HostSpace> field =
        pfc::data::field_from_subdomain<std::complex<double>>(decomp, rank, halo);

    field.apply([](double x, double y, double z) -> std::complex<double> {
      return std::complex<double>(x, y);
    });

    REQUIRE(field(1, 2, 3).real() == 1.0);
    REQUIRE(field(1, 2, 3).imag() == 2.0);
  }
}

TEST_CASE("Field indexing and element access", "[field][indexing]") {
  const int nx = 5, ny = 5, nz = 5;
  const pfc::Domain domain = pfc::domain::create({nx, ny, nz});
  const int num_ranks = 1;
  const pfc::Decomposition decomp = pfc::decomposition::create(domain, num_ranks);
  const int rank = 0;
  const int halo = 0;

  SECTION("Index operator writes and reads correctly") {
    pfc::data::Field<double, pfc::HostSpace> field =
        pfc::data::field_from_subdomain<double>(decomp, rank, halo);

    field(1, 2, 3) = 42.0;
    field(0, 0, 0) = 1.0;
    field(4, 4, 4) = 99.0;

    REQUIRE(field(1, 2, 3) == 42.0);
    REQUIRE(field(0, 0, 0) == 1.0);
    REQUIRE(field(4, 4, 4) == 99.0);
    REQUIRE(field(0, 0, 1) == 0.0); // Default initialized
  }

  SECTION("Index with Int3 coordinate") {
    pfc::data::Field<double, pfc::HostSpace> field =
        pfc::data::field_from_subdomain<double>(decomp, rank, halo);

    pfc::Int3 idx{2, 3, 1};
    field(idx) = 7.5;

    REQUIRE(field(2, 3, 1) == 7.5);
    REQUIRE(field(idx) == 7.5);
  }

  SECTION("Default initialization is zero") {
    pfc::data::Field<double, pfc::HostSpace> field =
        pfc::data::field_from_subdomain<double>(decomp, rank, halo);

    bool all_zero = true;
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          all_zero &= (field(i, j, k) == 0.0);
        }
      }
    }
    REQUIRE(all_zero);
  }
}

TEST_CASE("Field coordinate round-trip", "[field][coordinates]") {
  const int nx = 10, ny = 15, nz = 20;
  const pfc::Real3 spacing{2.0, 3.0, 4.0};
  const pfc::Real3 origin{1.0, 2.0, 3.0};
  const pfc::Domain domain = pfc::domain::create(pfc::GridSize({nx, ny, nz}),
                                                   pfc::PhysicalOrigin(origin),
                                                   pfc::GridSpacing(spacing));
  const int num_ranks = 1;
  const pfc::Decomposition decomp = pfc::decomposition::create(domain, num_ranks);
  const int rank = 0;
  const int halo = 0;

  SECTION("Local indices map to correct physical coordinates") {
    pfc::data::Field<double, pfc::HostSpace> field =
        pfc::data::field_from_subdomain<double>(decomp, rank, halo);

    // Test various local indices
    pfc::Real3 coords = field.coords(0, 0, 0);
    REQUIRE(coords[0] == origin[0]);
    REQUIRE(coords[1] == origin[1]);
    REQUIRE(coords[2] == origin[2]);

    coords = field.coords(1, 0, 0);
    REQUIRE(coords[0] == origin[0] + spacing[0]);
    REQUIRE(coords[1] == origin[1]);
    REQUIRE(coords[2] == origin[2]);

    coords = field.coords(2, 3, 4);
    REQUIRE(coords[0] == origin[0] + 2 * spacing[0]);
    REQUIRE(coords[1] == origin[1] + 3 * spacing[1]);
    REQUIRE(coords[2] == origin[2] + 4 * spacing[2]);
  }

  SECTION("Round-trip: indices to coords and back") {
    pfc::data::Field<double, pfc::HostSpace> field =
        pfc::data::field_from_subdomain<double>(decomp, rank, halo);

    // Store values at specific indices
    field(2, 3, 4) = 123.45;
    field(7, 8, 9) = 67.89;

    // Get coordinates for those indices
    pfc::Real3 coords_234 = field.coords(2, 3, 4);
    pfc::Real3 coords_789 = field.coords(7, 8, 9);

    // Verify expected coordinates
    REQUIRE(coords_234[0] == origin[0] + 2 * spacing[0]);
    REQUIRE(coords_234[1] == origin[1] + 3 * spacing[1]);
    REQUIRE(coords_234[2] == origin[2] + 4 * spacing[2]);

    REQUIRE(coords_789[0] == origin[0] + 7 * spacing[0]);
    REQUIRE(coords_789[1] == origin[1] + 8 * spacing[1]);
    REQUIRE(coords_789[2] == origin[2] + 9 * spacing[2]);

    // Verify we can still access the stored values
    REQUIRE(field(2, 3, 4) == 123.45);
    REQUIRE(field(7, 8, 9) == 67.89);
  }

  SECTION("Unit spacing and origin work correctly") {
    const pfc::Domain unit_domain = pfc::domain::create({nx, ny, nz});
    const pfc::Decomposition unit_decomp =
        pfc::decomposition::create(unit_domain, num_ranks);

    pfc::data::Field<double, pfc::HostSpace> field =
        pfc::data::field_from_subdomain<double>(unit_decomp, rank, halo);

    for (int i = 0; i < nx; ++i) {
      for (int j = 0; j < ny; ++j) {
        for (int k = 0; k < nz; ++k) {
          pfc::Real3 coords = field.coords(i, j, k);
          REQUIRE(coords[0] == i);
          REQUIRE(coords[1] == j);
          REQUIRE(coords[2] == k);
        }
      }
    }
  }

  SECTION("Apply function uses correct coordinates") {
    pfc::data::Field<double, pfc::HostSpace> field =
        pfc::data::field_from_subdomain<double>(decomp, rank, halo);

    // Apply a function that depends on coordinates
    field.apply([](double x, double y, double z) {
      return x * y * z;
    });

    // Manually compute expected value and compare
    pfc::Real3 coords_234 = field.coords(2, 3, 4);
    double expected_234 = coords_234[0] * coords_234[1] * coords_234[2];
    REQUIRE_THAT(field(2, 3, 4),
                  WithinAbs(expected_234, 1e-10));
  }
}

TEST_CASE("Field global index mapping", "[field][coordinates]") {
  const int nx = 8, ny = 6, nz = 4;
  const pfc::Domain domain = pfc::domain::create({nx, ny, nz});
  const int num_ranks = 1;
  const pfc::Decomposition decomp = pfc::decomposition::create(domain, num_ranks);
  const int rank = 0;
  const int halo = 0;

  SECTION("Single-rank global mapping is identity") {
    pfc::data::Field<double, pfc::HostSpace> field =
        pfc::data::field_from_subdomain<double>(decomp, rank, halo);

    for (int i = 0; i < nx; ++i) {
      for (int j = 0; j < ny; ++j) {
        for (int k = 0; k < nz; ++k) {
          pfc::Int3 global_idx = field.global(i, j, k);
          REQUIRE(global_idx[0] == i);
          REQUIRE(global_idx[1] == j);
          REQUIRE(global_idx[2] == k);
        }
      }
    }
  }
}

TEST_CASE("Field with different element types", "[field][types]") {
  const int nx = 3, ny = 3, nz = 3;
  const pfc::Domain domain = pfc::domain::create({nx, ny, nz});
  const int num_ranks = 1;
  const pfc::Decomposition decomp = pfc::decomposition::create(domain, num_ranks);
  const int rank = 0;
  const int halo = 0;

  SECTION("Field with int type") {
    pfc::data::Field<int, pfc::HostSpace> field =
        pfc::data::field_from_subdomain<int>(decomp, rank, halo);

    field(1, 1, 1) = 42;
    REQUIRE(field(1, 1, 1) == 42);
    REQUIRE(field(0, 0, 0) == 0);
  }

  SECTION("Field with std::complex type") {
    pfc::data::Field<std::complex<double>, pfc::HostSpace> field =
        pfc::data::field_from_subdomain<std::complex<double>>(decomp, rank, halo);

    field(1, 2, 1) = std::complex<double>(3.0, 4.0);
    REQUIRE(field(1, 2, 1).real() == 3.0);
    REQUIRE(field(1, 2, 1).imag() == 4.0);
  }

  SECTION("Apply function works with complex type") {
    pfc::data::Field<std::complex<double>, pfc::HostSpace> field =
        pfc::data::field_from_subdomain<std::complex<double>>(decomp, rank, halo);

    field.apply([](double x, double y, double z) -> std::complex<double> {
      return std::exp(std::complex<double>(0, x));
    });

    std::complex<double> val = field(2, 0, 0);
    REQUIRE_THAT(val.real(), WithinAbs(std::cos(2.0), 1e-10));
    REQUIRE_THAT(val.imag(), WithinAbs(std::sin(2.0), 1e-10));
  }
}
