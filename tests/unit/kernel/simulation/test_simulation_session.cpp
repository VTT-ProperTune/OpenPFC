// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <stdexcept>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/simulation/simulation_session.hpp>

using pfc::sim::SessionSelection;
using pfc::sim::SimulationBackend;
using pfc::sim::SimulationMethod;
using pfc::sim::stacks::FDCPUStack;
using pfc::sim::stacks::SpectralCPUStack;

namespace {

pfc::Domain tiny_domain() {
  return pfc::domain::create(pfc::GridSize({8, 8, 8}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({1.0, 1.0, 1.0}));
}

pfc::Time two_steps() { return pfc::Time({0.0, 0.2, 0.1}, 0.1); }

} // namespace

TEST_CASE("SimulationSession spectral cpu stack name and Time loop",
          "[simulation_session][unit]") {
  SessionSelection s{};
  pfc::sim::SimulationSession<SpectralCPUStack> session(s, tiny_domain(),
                                                        two_steps(), 0, 1);
  REQUIRE(std::string(session.stack_name()) == "SpectralCPUStack");
  REQUIRE(std::string(pfc::sim::intended_stack_name(session.selection())) ==
          session.stack_name());
  REQUIRE(session.stack().fft().size_inbox() > 0);
  int steps = 0;
  session.run([&](double) { ++steps; });
  REQUIRE(steps == 2);
}

TEST_CASE("SimulationSession fd cpu uses fd_order halo",
          "[simulation_session][unit]") {
  SessionSelection s{SimulationMethod::Fd, SimulationBackend::Cpu, 8};
  pfc::sim::SimulationSession<FDCPUStack> session(s, tiny_domain(), two_steps(), 0,
                                                  1);
  REQUIRE(std::string(session.stack_name()) == "FDCPUStack");
  REQUIRE(session.stack().fd_order() == 8);
  REQUIRE(session.stack().halo_width() == 4);
}

TEST_CASE("SimulationSession rejects a mismatched stack type",
          "[simulation_session][unit]") {
  SessionSelection fd{SimulationMethod::Fd, SimulationBackend::Cpu, 2};
  REQUIRE_THROWS_AS((pfc::sim::SimulationSession<SpectralCPUStack>(
                        fd, tiny_domain(), two_steps(), 0, 1)),
                    std::invalid_argument);
}
