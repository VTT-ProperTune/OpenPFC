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
#include "openpfc/initial_conditions/random_seeds.hpp"
#include "openpfc/model.hpp"

using namespace pfc;
using Catch::Approx;
using pfc::types::Int3;

// Mock model class for testing
class ModelWithRandomSeedsIC : public Model {
public:
  ModelWithRandomSeedsIC(FFT &fft, const pfc::World &world)
      : pfc::Model(fft, world) {}

  void step(double /*t*/) override {}
  void initialize(double /*dt*/) override {}
};

TEST_CASE("RandomSeeds - Parameter Access", "[ic_random_seeds]") {
  RandomSeeds seeds;

  SECTION("Set and get amplitude") {
    seeds.set_amplitude(0.25);
    REQUIRE(seeds.get_amplitude() == Approx(0.25));
  }

  SECTION("Set and get density") {
    seeds.set_density(0.55);
    REQUIRE(seeds.get_density() == Approx(0.55));
  }
}

TEST_CASE("RandomSeeds - Field Application", "[ic_random_seeds]") {
  // Create domain matching hardcoded values in RandomSeeds
  auto world =
      world::create(GridSize({32, 32, 32}), PhysicalOrigin({-128.0, -128.0, -128.0}),
                    GridSpacing({8.0, 8.0, 8.0}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  ModelWithRandomSeedsIC m(fft, world);

  const size_t field_size = fft.size_inbox();
  std::vector<double> psi(field_size, 0.0);
  m.add_real_field("default", psi);

  RandomSeeds seeds;
  seeds.set_amplitude(0.2);
  seeds.set_density(0.5);

  SECTION("Apply to field") {
    seeds.apply(m, 0.0);
    const Field &field = m.get_real_field("default");

    // Check that field has been modified (some seeds should be present)
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
    seeds.apply(m, 0.0);
    const Field &field = m.get_real_field("default");

    // Seeds use same formula as SingleSeed: rho + 2*amp*sum(cos(q_i . r))
    double max_expected = seeds.get_density() + 12.0 * seeds.get_amplitude();
    double min_expected = seeds.get_density() - 12.0 * seeds.get_amplitude();

    for (const auto &value : field) {
      if (value != 0.0) {                     // Inside a seed
        REQUIRE(value >= min_expected - 0.1); // Small tolerance
        REQUIRE(value <= max_expected + 0.1);
      }
    }
  }

  SECTION("Deterministic with fixed seed") {
    // RandomSeeds uses fixed random seed (42), so results should be deterministic
    seeds.apply(m, 0.0);
    Field field1 = m.get_real_field("default");

    // Reset field to zero
    std::vector<double> psi2(field_size, 0.0);
    m.add_real_field("default", psi2);

    // Apply again
    seeds.apply(m, 0.0);
    Field field2 = m.get_real_field("default");

    // Results should be identical
    REQUIRE(field1.size() == field2.size());
    for (size_t i = 0; i < field1.size(); ++i) {
      REQUIRE(field1[i] == Approx(field2[i]));
    }
  }
}

TEST_CASE("RandomSeeds - Integration with Model", "[ic_random_seeds]") {
  auto world =
      world::create(GridSize({16, 16, 16}), PhysicalOrigin({-128.0, -128.0, -128.0}),
                    GridSpacing({16.0, 16.0, 16.0}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  ModelWithRandomSeedsIC model(fft, world);

  const size_t field_size = fft.size_inbox();
  std::vector<double> psi(field_size, 0.0);
  model.add_real_field("density", psi);

  RandomSeeds seeds;
  seeds.set_field_name("density");
  seeds.set_amplitude(0.15);
  seeds.set_density(0.65);

  REQUIRE_NOTHROW(seeds.apply(model, 0.0));

  const Field &field = model.get_real_field("density");
  REQUIRE(field.size() == field_size);
}

TEST_CASE("RandomSeeds - Field Name Assignment", "[ic_random_seeds]") {
  RandomSeeds seeds;
  seeds.set_field_name("crystalline_phase");
  REQUIRE(seeds.get_field_name() == "crystalline_phase");
}
