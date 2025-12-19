// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>
#include <iostream>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "openpfc/core/decomposition.hpp"
#include "openpfc/core/types.hpp"
#include "openpfc/core/world.hpp"
#include "openpfc/factory/decomposition_factory.hpp"
#include "openpfc/fft.hpp"
#include "openpfc/initial_conditions/single_seed.hpp"
#include "openpfc/model.hpp"

using namespace pfc;
using Catch::Approx;
using pfc::types::Int3;

// Mock model class for testing
class ModelWithSingleSeedIC : public Model {
public:
  ModelWithSingleSeedIC(FFT &fft, const pfc::World &world)
      : pfc::Model(fft, world) {}

  void step(double /*t*/) override {}
  void initialize(double /*dt*/) override {}
};

TEST_CASE("SingleSeed - Parameter Access", "[ic_single_seed]") {
  SingleSeed seed;

  SECTION("Default values") {
    // SingleSeed doesn't initialize member variables, so we test setters/getters
  }

  SECTION("Set and get amplitude") {
    seed.set_amplitude(0.3);
    REQUIRE(seed.get_amplitude() == Approx(0.3));
  }

  SECTION("Set and get density") {
    seed.set_density(0.7);
    REQUIRE(seed.get_density() == Approx(0.7));
  }
}

TEST_CASE("SingleSeed - Field Application", "[ic_single_seed]") {
  // Create domain centered at origin for seed placement
  auto world =
      world::create(GridSize({16, 16, 16}), PhysicalOrigin({-128.0, -128.0, -128.0}),
                    GridSpacing({16.0, 16.0, 16.0}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  ModelWithSingleSeedIC m(fft, world);

  const size_t field_size = fft.size_inbox();
  std::vector<double> psi(field_size, 0.0);
  m.add_real_field("default", psi);

  SingleSeed seed;
  seed.set_amplitude(0.2);
  seed.set_density(0.5);

  SECTION("Apply to field") {
    seed.apply(m, 0.0);
    const Field &field = m.get_real_field("default");

    // Check that field has been modified
    bool has_nonzero = false;
    for (const auto &value : field) {
      if (value != 0.0) {
        has_nonzero = true;
        break;
      }
    }
    REQUIRE(has_nonzero);

    // The seed is centered at origin with radius 64.0
    // Points inside should have non-zero values
    // Points outside should remain zero
  }

  SECTION("Field values inside seed") {
    seed.apply(m, 0.0);
    const Field &field = m.get_real_field("default");

    // Check that values inside the seed are reasonable
    // SingleSeed uses: u = rho_seed + 2*amp*sum(cos(q_i . r))
    // With 6 modes, maximum is rho_seed + 12*amp
    // Minimum is rho_seed - 12*amp
    double max_expected = seed.get_density() + 12.0 * seed.get_amplitude();
    double min_expected = seed.get_density() - 12.0 * seed.get_amplitude();

    for (const auto &value : field) {
      if (value != 0.0) { // Inside seed
        REQUIRE(value >= min_expected);
        REQUIRE(value <= max_expected);
      }
    }
  }
}

TEST_CASE("SingleSeed - Integration with Model", "[ic_single_seed]") {
  auto world =
      world::create(GridSize({8, 8, 8}), PhysicalOrigin({-64.0, -64.0, -64.0}),
                    GridSpacing({16.0, 16.0, 16.0}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  ModelWithSingleSeedIC model(fft, world);

  const size_t field_size = fft.size_inbox();
  std::vector<double> psi(field_size, 0.0);
  model.add_real_field("density", psi);

  SingleSeed seed;
  seed.set_field_name("density");
  seed.set_amplitude(0.1);
  seed.set_density(0.6);

  REQUIRE_NOTHROW(seed.apply(model, 0.0));

  const Field &field = model.get_real_field("density");
  REQUIRE(field.size() == field_size);
}

TEST_CASE("SingleSeed - Field Name Assignment", "[ic_single_seed]") {
  SingleSeed seed;
  seed.set_field_name("custom_field");
  REQUIRE(seed.get_field_name() == "custom_field");
}
