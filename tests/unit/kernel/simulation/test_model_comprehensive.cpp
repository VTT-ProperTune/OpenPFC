// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/simulation/model.hpp>

#include "fixtures/simulation_factories.hpp"

using namespace pfc;

namespace {
class StubModel : public Model {
public:
  StubModel(FFT &fft, const World &world) : Model(fft, world) {}
  void step(double /*t*/) override {}
  void initialize(double /*dt*/) override {}
};
} // namespace

// Test fixture for comprehensive tests
class ComprehensiveModelFixture : public pfc::test::SimulationModelFixture {
public:
  ComprehensiveModelFixture() {
    SetUpDefaultDomain(8, 1, 1);
  }
};

TEST_CASE_METHOD(ComprehensiveModelFixture, "Model - comprehensive (stub)", "[model][comprehensive][unit]") {
  World world = pfc::test::world_from_domain(domain());
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  StubModel model(fft, world);
  REQUIRE(get_size(get_world(model)) == pfc::types::Int3{8, 1, 1});
}
