// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <string>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json_simulation_session.hpp>
#include <openpfc/kernel/simulation/stacks/fd_cpu_stack.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

using nlohmann::json;
using pfc::sim::stacks::FDCPUStack;
using pfc::sim::stacks::SpectralCPUStack;

namespace {

json matrix_doc(const char *method, const char *backend, int fd_order = 2) {
  return json{
      {"method", method},
      {"backend", backend},
      {"fd_order", fd_order},
      {"Lx", 8},
      {"Ly", 8},
      {"Lz", 8},
      {"dx", 1.0},
      {"dy", 1.0},
      {"dz", 1.0},
      {"origin", "corner"},
      {"timestepping", {{"t0", 0.0}, {"t1", 0.2}, {"dt", 0.1}, {"saveat", 0.1}}}};
}

} // namespace

TEST_CASE("session matrix spectral cpu from JSON", "[session_matrix][cpu]") {
  const json doc = matrix_doc("spectral", "cpu");
  auto session = pfc::ui::make_simulation_session<SpectralCPUStack>(doc, 0, 1);
  REQUIRE(std::string(session.stack_name()) ==
          std::string(pfc::sim::intended_stack_name(session.selection())));
  int steps = 0;
  session.run([&](double) { ++steps; });
  REQUIRE(steps == 2);
}

TEST_CASE("session matrix fd cpu from JSON", "[session_matrix][cpu]") {
  const json doc = matrix_doc("fd", "cpu", 4);
  auto session = pfc::ui::make_simulation_session<FDCPUStack>(doc, 0, 1);
  REQUIRE(std::string(session.stack_name()) == "FDCPUStack");
  REQUIRE(session.stack().fd_order() == 4);
  int steps = 0;
  session.run([&](double) { ++steps; });
  REQUIRE(steps == 2);
}

TEST_CASE("session matrix fftw alias selects cpu spectral stack",
          "[session_matrix][cpu]") {
  const json doc = matrix_doc("spectral", "fftw");
  auto session = pfc::ui::make_simulation_session<SpectralCPUStack>(doc, 0, 1);
  REQUIRE(std::string(session.stack_name()) == "SpectralCPUStack");
}
