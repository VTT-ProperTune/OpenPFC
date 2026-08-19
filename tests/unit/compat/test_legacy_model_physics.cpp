// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mpi.h>

#include <fixtures/diffusion_model.hpp>
#include <fixtures/mock_model.hpp>
#include <fixtures/simulation_factories.hpp>

#include <openpfc/compat/legacy_model_physics.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulator.hpp>
#include <openpfc/kernel/simulation/time.hpp>

using Catch::Matchers::WithinAbs;
using pfc::Simulator;
using pfc::Time;
using pfc::compat::LegacyModelPhysics;
using pfc::sim::SteppablePhysics;

TEST_CASE("A2 step_with_physics plus A1 matches Simulator::step",
          "[a2][a1][adapter][unit]") {
  auto domain = pfc::test::DomainFactory::create_default_domain(8, 8, 8);
  auto world = pfc::test::world_from_domain(domain);
  auto decomp = pfc::decomposition::create(world, 1);
  auto fft = pfc::fft::create(decomp);

  pfc::testing::InstrumentedMockModel model_a(fft, world);
  pfc::testing::InstrumentedMockModel model_b(fft, world);
  Time time_a({0.0, 1.5, 0.5}, 0.0);
  Time time_b({0.0, 1.5, 0.5}, 0.0);
  Simulator sim_a(model_a, time_a);
  Simulator sim_b(model_b, time_b);
  pfc::initialize(sim_a);
  pfc::initialize(sim_b);

  LegacyModelPhysics physics(model_b);
  static_assert(SteppablePhysics<LegacyModelPhysics>);

  while (!pfc::done(sim_a)) {
    pfc::step(sim_a);
  }
  while (!pfc::done(sim_b)) {
    sim_b.step_with_physics([&] {
      physics.step(pfc::time::current(sim_b.get_time()));
    });
  }

  REQUIRE(model_a.step_call_count == model_b.step_call_count);
  REQUIRE(model_a.step_call_count == 3);
  REQUIRE_THAT(model_b.last_step_time, WithinAbs(1.5, 1e-10));
  REQUIRE_THAT(model_a.last_step_time, WithinAbs(model_b.last_step_time, 0.0));
}

TEST_CASE("A1 diffusion fixture: Simulator vs LegacyModelPhysics bitwise",
          "[a1][adapter][unit][diffusion]") {
  auto domain = pfc::test::DomainFactory::create_default_domain(8, 8, 8);
  auto world = pfc::test::world_from_domain(domain);
  auto decomp = pfc::decomposition::create(world, 1);
  auto fft = pfc::fft::create(decomp);

  pfc::test::DiffusionModel model_a(fft, world);
  pfc::test::DiffusionModel model_b(fft, world);
  Time time_a({0.0, 0.03, 0.01}, 0.0);
  Time time_b({0.0, 0.03, 0.01}, 0.0);
  Simulator sim_a(model_a, time_a);
  Simulator sim_b(model_b, time_b);
  pfc::initialize(sim_a);
  pfc::initialize(sim_b);

  REQUIRE(model_a.m_psi == model_b.m_psi);

  LegacyModelPhysics physics(model_b);
  while (!pfc::done(sim_a)) {
    pfc::step(sim_a);
  }
  while (!pfc::done(sim_b)) {
    sim_b.step_with_physics([&] {
      physics.step(pfc::time::current(sim_b.get_time()));
    });
  }

  REQUIRE(model_a.m_psi.size() == model_b.m_psi.size());
  REQUIRE(model_a.m_psi == model_b.m_psi);
}
