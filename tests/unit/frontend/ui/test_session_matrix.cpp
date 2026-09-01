// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>
#include <stdexcept>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json_simulation_session.hpp>
#include <openpfc/frontend/ui/json_fd_session.hpp>
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

TEST_CASE("session matrix fd cpu JSON runs composed Euler heat",
          "[session_matrix][cpu][fd]") {
  json doc = matrix_doc("fd", "cpu", 2);
  doc["timestepping"]["integrator"] = json{{"method", "euler"}};

  auto session = pfc::ui::make_simulation_session<FDCPUStack>(doc, 0, 1);
  pfc::ui::overlay_simulator_integrator_method(session.time(), doc);
  pfc::ui::fill_fd_sine_x(session.stack().u());
  // Interior of an order-2 Laplacian is [hw, n-hw) per axis; (1,0,0) is a
  // y/z face cell and is not written.
  const double before = std::abs(session.stack().u()(3, 3, 3));
  REQUIRE(before > 1e-12);
  const int steps = pfc::ui::step_fd_cpu_session(session);
  REQUIRE(steps == 2);
  REQUIRE(std::abs(session.stack().u()(3, 3, 3)) < before);

  REQUIRE(pfc::ui::run_fd_cpu_json_session(doc, 0, 1) == 2);
}

TEST_CASE("JSON FD session rejects etd1 (use compose_etd1)",
          "[session_matrix][cpu][fd]") {
  json doc = matrix_doc("fd", "cpu", 2);
  doc["timestepping"]["integrator"] = json{{"method", "etd1"}};
  REQUIRE_THROWS_AS(pfc::ui::run_fd_cpu_json_session(doc, 0, 1),
                    std::invalid_argument);
}

TEST_CASE("session matrix fftw alias selects cpu spectral stack",
          "[session_matrix][cpu]") {
  const json doc = matrix_doc("spectral", "fftw");
  auto session = pfc::ui::make_simulation_session<SpectralCPUStack>(doc, 0, 1);
  REQUIRE(std::string(session.stack_name()) == "SpectralCPUStack");
}
