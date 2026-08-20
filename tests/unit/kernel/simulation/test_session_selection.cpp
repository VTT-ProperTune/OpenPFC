// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <string>

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/simulation/session_selection.hpp>

using pfc::sim::SessionSelection;
using pfc::sim::SimulationBackend;
using pfc::sim::SimulationMethod;

TEST_CASE("simulation_method_from_string", "[session_selection][unit]") {
  REQUIRE(pfc::sim::simulation_method_from_string("spectral") ==
          SimulationMethod::Spectral);
  REQUIRE(pfc::sim::simulation_method_from_string("fd") == SimulationMethod::Fd);
  REQUIRE_FALSE(pfc::sim::simulation_method_from_string("euler").has_value());
  REQUIRE(std::string(pfc::sim::to_cstring(SimulationMethod::Spectral)) ==
          "spectral");
  REQUIRE(std::string(pfc::sim::to_cstring(SimulationMethod::Fd)) == "fd");
}

TEST_CASE("simulation_backend_from_string aliases fftw and rocm",
          "[session_selection][unit]") {
  REQUIRE(pfc::sim::simulation_backend_from_string("cpu") == SimulationBackend::Cpu);
  REQUIRE(pfc::sim::simulation_backend_from_string("fftw") ==
          SimulationBackend::Cpu);
  REQUIRE(pfc::sim::simulation_backend_from_string("cuda") ==
          SimulationBackend::Cuda);
  REQUIRE(pfc::sim::simulation_backend_from_string("hip") == SimulationBackend::Hip);
  REQUIRE(pfc::sim::simulation_backend_from_string("rocm") ==
          SimulationBackend::Hip);
  REQUIRE_FALSE(pfc::sim::simulation_backend_from_string("opencl").has_value());
  REQUIRE(std::string(pfc::sim::to_cstring(SimulationBackend::Cpu)) == "cpu");
}

TEST_CASE("even_fd_order and halo width", "[session_selection][unit]") {
  REQUIRE(pfc::sim::even_fd_order(2));
  REQUIRE(pfc::sim::even_fd_order(20));
  REQUIRE_FALSE(pfc::sim::even_fd_order(1));
  REQUIRE_FALSE(pfc::sim::even_fd_order(3));
  REQUIRE_FALSE(pfc::sim::even_fd_order(22));
  REQUIRE(pfc::sim::halo_width_from_fd_order(4) == 2);
}

TEST_CASE("intended_stack_name for method x backend", "[session_selection][unit]") {
  SessionSelection cpu_spec{};
  REQUIRE(std::string(pfc::sim::intended_stack_name(cpu_spec)) ==
          "SpectralCPUStack");
  SessionSelection fd_cpu{SimulationMethod::Fd, SimulationBackend::Cpu, 2};
  REQUIRE(std::string(pfc::sim::intended_stack_name(fd_cpu)) == "FDCPUStack");
  SessionSelection spec_cuda{SimulationMethod::Spectral, SimulationBackend::Cuda, 2};
  REQUIRE(std::string(pfc::sim::intended_stack_name(spec_cuda)) ==
          "GPUSpectralStack<CUDASpace>");
}

TEST_CASE("session_backend_compiled cpu is always true",
          "[session_selection][unit]") {
  REQUIRE(pfc::sim::session_backend_compiled(SimulationBackend::Cpu,
                                             SimulationMethod::Spectral));
  REQUIRE(pfc::sim::session_backend_compiled(SimulationBackend::Cpu,
                                             SimulationMethod::Fd));
}

TEST_CASE("require_session_for_stack accepts a matching compiled pair",
          "[session_selection][unit]") {
  SessionSelection s{};
  REQUIRE_NOTHROW(pfc::sim::require_session_for_stack(s, SimulationMethod::Spectral,
                                                      SimulationBackend::Cpu));
}
