// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <stdexcept>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/simulation/session_stack_factory.hpp>

using Catch::Matchers::ContainsSubstring;
using pfc::sim::SessionSelection;
using pfc::sim::SimulationBackend;
using pfc::sim::SimulationMethod;

namespace {

pfc::Domain tiny_domain() {
  return pfc::domain::create(pfc::GridSize({8, 8, 8}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({1.0, 1.0, 1.0}));
}

} // namespace

#ifdef OpenPFC_ENABLE_HEFFTE
TEST_CASE("make_spectral_cpu_stack from default SessionSelection",
          "[session_stack_factory][unit]") {
  SessionSelection s{};
  auto stack = pfc::sim::make_spectral_cpu_stack(s, tiny_domain(), 0, 1);
  REQUIRE(stack.rank() == 0);
  REQUIRE(stack.nproc() == 1);
  REQUIRE(stack.fft().size_inbox() > 0);
  REQUIRE(std::string(pfc::sim::intended_stack_name(s)) == "SpectralCPUStack");
}
#endif

TEST_CASE("make_fd_cpu_stack uses fd_order for halo width",
          "[session_stack_factory][unit]") {
  SessionSelection s{SimulationMethod::Fd, SimulationBackend::Cpu, 8};
  auto stack = pfc::sim::make_fd_cpu_stack(s, tiny_domain(), 0, 1);
  REQUIRE(stack.fd_order() == 8);
  REQUIRE(stack.halo_width() == 4);
}

TEST_CASE("make_fd_padded_cpu_stack uses fd_order/2 storage halo",
          "[session_stack_factory][unit]") {
  SessionSelection s{SimulationMethod::Fd, SimulationBackend::Cpu, 6};
  auto stack = pfc::sim::make_fd_padded_cpu_stack(s, tiny_domain(), 0, 1);
  REQUIRE(stack.halo_width() == 3);
}

#ifdef OpenPFC_ENABLE_HEFFTE
TEST_CASE("CPU stack factory rejects a mismatched method",
          "[session_stack_factory][unit]") {
  SessionSelection fd{SimulationMethod::Fd, SimulationBackend::Cpu, 2};
  REQUIRE_THROWS_AS(pfc::sim::make_spectral_cpu_stack(fd, tiny_domain(), 0, 1),
                    std::invalid_argument);
  try {
    (void)pfc::sim::make_spectral_cpu_stack(fd, tiny_domain(), 0, 1);
  } catch (const std::invalid_argument &e) {
    REQUIRE_THAT(std::string(e.what()), ContainsSubstring("FDCPUStack"));
  }
}

TEST_CASE("CPU stack factory rejects cuda SessionSelection",
          "[session_stack_factory][unit]") {
  SessionSelection cuda{SimulationMethod::Spectral, SimulationBackend::Cuda, 2};
  REQUIRE_THROWS_AS(pfc::sim::make_spectral_cpu_stack(cuda, tiny_domain(), 0, 1),
                    std::invalid_argument);
}
#endif

TEST_CASE("require_session_for_stack rejects odd fd_order",
          "[session_stack_factory][unit]") {
  SessionSelection odd{SimulationMethod::Fd, SimulationBackend::Cpu, 3};
  REQUIRE_THROWS_AS(pfc::sim::require_session_for_stack(odd, SimulationMethod::Fd,
                                                        SimulationBackend::Cpu),
                    std::invalid_argument);
}
