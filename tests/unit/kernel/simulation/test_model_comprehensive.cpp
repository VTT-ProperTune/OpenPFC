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
  StubModel(FFT &fft, const Domain &domain) : Model(fft, domain) {}
  void step(double /*t*/) override {}
  void initialize(double /*dt*/) override {}
};
} // namespace

// Test fixture for comprehensive tests
class ComprehensiveModelFixture : public pfc::test::SimulationModelFixture {
public:
  ComprehensiveModelFixture() { SetUpDefaultDomain(8, 1, 1); }
};

TEST_CASE_METHOD(ComprehensiveModelFixture, "Model - comprehensive (stub)",
                 "[model][comprehensive][unit]") {
  auto decomposition = decomposition::create(domain(), 1);
  auto fft = fft::create(decomposition);
  StubModel model(fft, domain());
  REQUIRE(get_domain(model).size == pfc::types::Int3{8, 1, 1});
}
