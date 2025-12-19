// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "openpfc/core/decomposition.hpp"
#include "openpfc/core/types.hpp"
#include "openpfc/core/world.hpp"
#include "openpfc/factory/decomposition_factory.hpp"
#include "openpfc/fft.hpp"
#include "openpfc/initial_conditions/seed_grid.hpp"
#include "openpfc/model.hpp"

using namespace pfc;
using Catch::Approx;
using pfc::types::Int3;

// Mock model class for testing
class ModelWithSeedGridIC : public Model {
public:
  ModelWithSeedGridIC(FFT &fft, const pfc::World &world) : pfc::Model(fft, world) {}

  void step(double /*t*/) override {}
  void initialize(double /*dt*/) override {}
};

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
  REQUIRE(grid.get_Nx() == 1); // Nx is always set to 1 in constructor
  REQUIRE(grid.get_Ny() == 3);
  REQUIRE(grid.get_Nz() == 4);
  REQUIRE(grid.get_X0() == Approx(100.0));
  REQUIRE(grid.get_radius() == Approx(15.0));
}

TEST_CASE("SeedGrid - Field Application", "[ic_seed_grid]") {
  auto world =
      world::create(GridSize({32, 32, 32}), PhysicalOrigin({-128.0, -128.0, -128.0}),
                    GridSpacing({8.0, 8.0, 8.0}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  ModelWithSeedGridIC m(fft, world);

  const size_t field_size = fft.size_inbox();
  std::vector<double> psi(field_size, 0.0);
  m.add_real_field("default", psi);

  SeedGrid grid;
  grid.set_Nx(1);
  grid.set_Ny(2);
  grid.set_Nz(2);
  grid.set_X0(-100.0);
  grid.set_radius(20.0);
  grid.set_density(0.5);
  grid.set_amplitude(0.2);

  SECTION("Apply to field") {
    grid.apply(m, 0.0);
    const Field &field = m.get_real_field("default");

    // Check that field has been modified (seeds should be present)
    bool has_nonzero = false;
    for (const auto &value : field) {
      if (value != 0.0) {
        has_nonzero = true;
        break;
      }
    }
    REQUIRE(has_nonzero);
  }

  SECTION("Field values in range") {
    grid.apply(m, 0.0);
    const Field &field = m.get_real_field("default");

    // Seeds use same formula: rho + 2*amp*sum(cos(q_i . r))
    double max_expected = grid.get_density() + 12.0 * grid.get_amplitude();
    double min_expected = grid.get_density() - 12.0 * grid.get_amplitude();

    for (const auto &value : field) {
      if (value != 0.0) {                     // Inside a seed
        REQUIRE(value >= min_expected - 0.1); // Small tolerance
        REQUIRE(value <= max_expected + 0.1);
      }
    }
  }

  SECTION("Deterministic with fixed seed") {
    // SeedGrid uses fixed random seed (42), so results should be deterministic
    grid.apply(m, 0.0);
    Field field1 = m.get_real_field("default");

    // Reset field to zero
    std::vector<double> psi2(field_size, 0.0);
    m.add_real_field("default", psi2);

    // Apply again
    grid.apply(m, 0.0);
    Field field2 = m.get_real_field("default");

    // Results should be identical
    REQUIRE(field1.size() == field2.size());
    for (size_t i = 0; i < field1.size(); ++i) {
      REQUIRE(field1[i] == Approx(field2[i]));
    }
  }
}

TEST_CASE("SeedGrid - Integration with Model", "[ic_seed_grid]") {
  auto world =
      world::create(GridSize({16, 16, 16}), PhysicalOrigin({-100.0, -100.0, -100.0}),
                    GridSpacing({12.5, 12.5, 12.5}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  ModelWithSeedGridIC model(fft, world);

  const size_t field_size = fft.size_inbox();
  std::vector<double> psi(field_size, 0.0);
  model.add_real_field("density", psi);

  SeedGrid grid;
  grid.set_field_name("density");
  grid.set_Ny(2);
  grid.set_Nz(2);
  grid.set_radius(15.0);
  grid.set_amplitude(0.1);
  grid.set_density(0.7);

  REQUIRE_NOTHROW(grid.apply(model, 0.0));

  const Field &field = model.get_real_field("density");
  REQUIRE(field.size() == field_size);
}

TEST_CASE("SeedGrid - Field Name Assignment", "[ic_seed_grid]") {
  SeedGrid grid;
  grid.set_field_name("grain_structure");
  REQUIRE(grid.get_field_name() == "grain_structure");
}
