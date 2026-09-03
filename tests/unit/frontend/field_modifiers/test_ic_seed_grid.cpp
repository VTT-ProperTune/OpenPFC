// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/simulation/initial_conditions/seed_grid.hpp>

using namespace pfc;
using Catch::Approx;

TEST_CASE("SeedGrid - Parameter Access", "[ic_seed_grid]") {
  SeedGrid grid;

  SECTION("Default values") {
    REQUIRE(grid.get_Nx() == 1);
    REQUIRE(grid.get_Ny() == 2);
    REQUIRE(grid.get_Nz() == 2);
  }

  SECTION("Set and get grid dimensions") {
    grid.set_Nx(3);
    grid.set_Ny(4);
    grid.set_Nz(5);
    REQUIRE(grid.get_Nx() == 3);
    REQUIRE(grid.get_Ny() == 4);
    REQUIRE(grid.get_Nz() == 5);
  }

  SECTION("Set and get radius") {
    grid.set_radius(10.0);
    REQUIRE(grid.get_radius() == Approx(10.0));
  }

  SECTION("Set and get density") {
    grid.set_density(0.6);
    REQUIRE(grid.get_density() == Approx(0.6));
  }

  SECTION("Set and get amplitude") {
    grid.set_amplitude(0.3);
    REQUIRE(grid.get_amplitude() == Approx(0.3));
  }

  SECTION("Set and get X0") {
    grid.set_X0(-50.0);
    REQUIRE(grid.get_X0() == Approx(-50.0));
  }
}

TEST_CASE("SeedGrid - Constructor with Parameters", "[ic_seed_grid]") {
  SeedGrid grid(3, 4, 100.0, 15.0);
  REQUIRE(grid.get_Nx() == 1);
  REQUIRE(grid.get_Ny() == 3);
  REQUIRE(grid.get_Nz() == 4);
  REQUIRE(grid.get_X0() == Approx(100.0));
  REQUIRE(grid.get_radius() == Approx(15.0));
}

TEST_CASE("SeedGrid - Field Application", "[ic_seed_grid]") {
  auto domain = pfc::domain::create(pfc::GridSize({32, 32, 32}),
                                    pfc::PhysicalOrigin({-128.0, -128.0, -128.0}),
                                    pfc::GridSpacing({8.0, 8.0, 8.0}));
  auto box = pfc::domain::index_box(domain);
  const size_t field_size = static_cast<size_t>(box.count());
  std::vector<double> psi(field_size, 0.0);

  SeedGrid grid;
  grid.set_Nx(1);
  grid.set_Ny(2);
  grid.set_Nz(2);
  grid.set_X0(-100.0);
  grid.set_radius(20.0);
  grid.set_density(0.5);
  grid.set_amplitude(0.2);

  SECTION("Apply to field") {
    grid.apply(psi, domain, box);
    bool has_nonzero = false;
    for (const auto &value : psi) {
      if (value != 0.0) {
        has_nonzero = true;
        break;
      }
    }
    REQUIRE(has_nonzero);
  }

  SECTION("Field values in range") {
    grid.apply(psi, domain, box);
    double max_expected = grid.get_density() + 12.0 * grid.get_amplitude();
    double min_expected = grid.get_density() - 12.0 * grid.get_amplitude();
    bool values_in_range = true;
    for (const auto &value : psi) {
      if (value != 0.0) {
        values_in_range &= value >= min_expected - 0.1;
        values_in_range &= value <= max_expected + 0.1;
      }
    }
    REQUIRE(values_in_range);
  }

  SECTION("Deterministic with fixed seed") {
    grid.apply(psi, domain, box);
    std::vector<double> field1 = psi;
    std::vector<double> psi2(field_size, 0.0);
    grid.apply(psi2, domain, box);
    REQUIRE(field1 == psi2);
  }
}

TEST_CASE("SeedGrid - Apply on named density field", "[ic_seed_grid]") {
  auto domain = pfc::domain::create(pfc::GridSize({16, 16, 16}),
                                    pfc::PhysicalOrigin({-100.0, -100.0, -100.0}),
                                    pfc::GridSpacing({12.5, 12.5, 12.5}));
  auto box = pfc::domain::index_box(domain);
  std::vector<double> psi(static_cast<size_t>(box.count()), 0.0);

  SeedGrid grid;
  grid.set_field_name("density");
  grid.set_Ny(2);
  grid.set_Nz(2);
  grid.set_radius(15.0);
  grid.set_amplitude(0.1);
  grid.set_density(0.7);

  REQUIRE_NOTHROW(grid.apply(psi, domain, box));
  REQUIRE(psi.size() == static_cast<size_t>(box.count()));
}

TEST_CASE("SeedGrid - Field Name Assignment", "[ic_seed_grid]") {
  SeedGrid grid;
  grid.set_field_name("grain_structure");
  REQUIRE(grid.get_field_name() == "grain_structure");
}
