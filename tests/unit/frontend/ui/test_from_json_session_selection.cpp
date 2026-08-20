// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <stdexcept>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json_session_selection.hpp>

using Catch::Matchers::ContainsSubstring;
using nlohmann::json;
using pfc::sim::SessionSelection;
using pfc::sim::SimulationBackend;
using pfc::sim::SimulationMethod;

TEST_CASE("from_json SessionSelection defaults to spectral cpu",
          "[ui][session_selection]") {
  const auto s = pfc::ui::from_json<SessionSelection>(json::object());
  REQUIRE(s.method == SimulationMethod::Spectral);
  REQUIRE(s.backend == SimulationBackend::Cpu);
  REQUIRE(s.fd_order == 2);
}

TEST_CASE("from_json SessionSelection maps fftw backend to cpu",
          "[ui][session_selection]") {
  const auto s = pfc::ui::from_json<SessionSelection>(json{{"backend", "fftw"}});
  REQUIRE(s.backend == SimulationBackend::Cpu);
}

TEST_CASE("from_json SessionSelection accepts method fd and fd_order",
          "[ui][session_selection]") {
  const auto s = pfc::ui::from_json<SessionSelection>(
      json{{"method", "fd"}, {"backend", "cpu"}, {"fd_order", 8}});
  REQUIRE(s.method == SimulationMethod::Fd);
  REQUIRE(s.backend == SimulationBackend::Cpu);
  REQUIRE(s.fd_order == 8);
  REQUIRE(std::string(pfc::sim::intended_stack_name(s)) == "FDCPUStack");
}

TEST_CASE("from_json SessionSelection rejects unknown method",
          "[ui][session_selection]") {
  REQUIRE_THROWS_AS(pfc::ui::from_json<SessionSelection>(json{{"method", "mesh"}}),
                    std::invalid_argument);
  try {
    (void)pfc::ui::from_json<SessionSelection>(json{{"method", "mesh"}});
  } catch (const std::invalid_argument &e) {
    REQUIRE_THAT(std::string(e.what()), ContainsSubstring("method"));
    REQUIRE_THAT(std::string(e.what()), ContainsSubstring("spectral"));
    REQUIRE_THAT(std::string(e.what()), ContainsSubstring("fd"));
  }
}

TEST_CASE("from_json SessionSelection rejects unknown backend",
          "[ui][session_selection]") {
  REQUIRE_THROWS_AS(
      pfc::ui::from_json<SessionSelection>(json{{"backend", "opencl"}}),
      std::invalid_argument);
  try {
    (void)pfc::ui::from_json<SessionSelection>(json{{"backend", "opencl"}});
  } catch (const std::invalid_argument &e) {
    REQUIRE_THAT(std::string(e.what()), ContainsSubstring("backend"));
    REQUIRE_THAT(std::string(e.what()), ContainsSubstring("cpu"));
  }
}

TEST_CASE("from_json SessionSelection rejects odd fd_order",
          "[ui][session_selection]") {
  REQUIRE_THROWS_AS(pfc::ui::from_json<SessionSelection>(json{{"fd_order", 3}}),
                    std::invalid_argument);
  try {
    (void)pfc::ui::from_json<SessionSelection>(json{{"fd_order", 3}});
  } catch (const std::invalid_argument &e) {
    REQUIRE_THAT(std::string(e.what()), ContainsSubstring("fd_order"));
    REQUIRE_THAT(std::string(e.what()), ContainsSubstring("2"));
    REQUIRE_THAT(std::string(e.what()), ContainsSubstring("20"));
  }
}

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
TEST_CASE("from_json SessionSelection maps cuda backend when compiled in",
          "[ui][session_selection]") {
  const auto s = pfc::ui::from_json<SessionSelection>(
      json{{"method", "spectral"}, {"backend", "cuda"}});
  REQUIRE(s.backend == SimulationBackend::Cuda);
  REQUIRE(std::string(pfc::sim::intended_stack_name(s)) ==
          "GPUSpectralStack<CUDASpace>");
}
#endif
