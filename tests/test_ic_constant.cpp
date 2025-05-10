// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/core/decomposition.hpp"
#include "openpfc/core/types.hpp"
#include "openpfc/core/world.hpp"
#include "openpfc/factory/decomposition_factory.hpp"
#include "openpfc/fft.hpp"
#include "openpfc/initial_conditions/constant.hpp"
#include "openpfc/model.hpp"
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <vector>

using namespace pfc;
using pfc::types::Int3;

// Mock model class for testing
class ModelWithConstantIC : public Model {
public:
  ModelWithConstantIC(const pfc::World &world) : pfc::Model(world) {}

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
    auto world = world::create({8, 1, 1});
    auto decomposition = make_decomposition(world, 0, 1);
    auto fft = fft::create(decomposition);
    ModelWithConstantIC m(world);
    m.set_fft(fft); // Ensure FFT object is set
    std::vector<double> psi(8);
    m.add_real_field("default", psi);

    Constant c(1.0);
    c.apply(m, 0.0);
    const Field &field = m.get_real_field("default");
    for (const auto &value : field) {
      REQUIRE(value == 1.0);
    }
  }
}

TEST_CASE("IC Constant - FFT Integration", "[ic_constant]") {
  auto world = world::create({8, 8, 8});
  auto decomposition = make_decomposition(world, 0, 1);
  auto fft = fft::create(decomposition);

  REQUIRE(fft.size_inbox() > 0);
  REQUIRE(fft.size_outbox() > 0);
  REQUIRE(fft.size_workspace() > 0);
}

TEST_CASE("IC Constant - Model Integration", "[ic_constant]") {
  auto world = world::create({8, 8, 8});
  auto decomposition = make_decomposition(world, 0, 1);
  auto fft = fft::create(decomposition);
  ModelWithConstantIC model(world);
  model.set_fft(fft);

  REQUIRE(get_size(model.get_world()) == Int3{8, 8, 8});
}
