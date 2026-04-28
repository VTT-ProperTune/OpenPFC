// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_types.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/simulation/initial_conditions/constant.hpp>
#include <openpfc/kernel/simulation/model.hpp>

using namespace pfc;
using pfc::types::Int3;

// Mock model class for testing
class ModelWithConstantIC : public Model {
public:
  ModelWithConstantIC(FFT &fft, const pfc::World &world) : pfc::Model(fft, world) {}

  void step(double /*t*/) override {}        // Suppress unused parameter warning
  void initialize(double /*dt*/) override {} // Suppress unused parameter warning
};

TEST_CASE("Constant Field Modifier") {

  SECTION("Density value") {
    Constant c(1.0);
    REQUIRE(c.get_density() == 1.0);
    c.set_density(2.5);
    REQUIRE(c.get_density() == 2.5);
  }

  SECTION("Apply field modifier") {
    auto world = world::create(GridSize({8, 1, 1}));
    auto decomposition = decomposition::create(world, 1);
    auto fft = fft::create(decomposition);
    ModelWithConstantIC m(fft, world);
    std::vector<double> psi(8);
    add_real_field(m, "default", psi);

    Constant c(1.0);
    c.apply(m, 0.0);
    const Field &field = m.get_real_field("default");
    for (const auto &value : field) {
      REQUIRE(value == 1.0);
    }
  }
}

TEST_CASE("IC Constant - FFT Integration", "[ic_constant]") {
  auto world = world::create(GridSize({8, 8, 8}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);

  REQUIRE(fft.size_inbox() > 0);
  REQUIRE(fft.size_outbox() > 0);
  REQUIRE(fft.size_workspace() > 0);
}

TEST_CASE("IC Constant - Model Integration", "[ic_constant]") {
  auto world = world::create(GridSize({8, 8, 8}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  ModelWithConstantIC model(fft, world);

  REQUIRE(get_size(get_world(model)) == Int3{8, 8, 8});
}
