// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <stdexcept>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/ui/from_json_log.hpp>
#include <openpfc/frontend/ui/json_checkpoint.hpp>
#include <openpfc/frontend/ui/json_driver_hooks.hpp>
#include <openpfc/frontend/ui/json_fd_session.hpp>
#include <openpfc/frontend/ui/results_writer_catalog.hpp>
#include <openpfc/kernel/simulation/time.hpp>

TEST_CASE("configure_json_driver_hooks sets from_json log rank", "[ui][hooks]") {
  pfc::ui::configure_json_driver_hooks(MPI_COMM_WORLD, 41);
  REQUIRE(pfc::ui::get_from_json_log_rank() == 41);
  pfc::ui::configure_json_driver_hooks(MPI_COMM_WORLD, -1);
  REQUIRE(pfc::ui::get_from_json_log_rank() == -1);
}

TEST_CASE("unknown results writer type is a hard error", "[ui][writers]") {
  REQUIRE_THROWS_AS(
      pfc::ui::default_results_writer_catalog().create_writer(
          "nope", "out.bin", MPI_COMM_WORLD, "psi"),
      std::invalid_argument);
  try {
    pfc::ui::default_results_writer_catalog().create_writer(
        "nope", "out.bin", MPI_COMM_WORLD, "psi");
  } catch (const std::invalid_argument &e) {
    REQUIRE_THAT(std::string(e.what()),
                 Catch::Matchers::ContainsSubstring("writer"));
    REQUIRE_THAT(std::string(e.what()),
                 Catch::Matchers::ContainsSubstring("binary"));
  }
}

TEST_CASE("unknown field-modifier type is a hard error", "[ui][modifiers]") {
  REQUIRE_THROWS_AS(pfc::ui::default_field_modifier_catalog().create_modifier(
                        "not_a_type", nlohmann::json::object()),
                    std::invalid_argument);
}

TEST_CASE("overlay_simulator_integrator_method overlays Time method",
          "[ui][integrator]") {
  pfc::Time time({0.0, 1.0, 0.1}, 1.0);
  const nlohmann::json settings = {
      {"simulator", {{"integrator", {{"method", "rk4_classical"}}}}}};
  pfc::ui::overlay_simulator_integrator_method(time, settings);
  REQUIRE(time.method() == pfc::sim::steppers::RKIntegratorMethod::RK4_Classical);

  const nlohmann::json etd = {{"simulator", {{"integrator", {{"method", "etd1"}}}}}};
  pfc::ui::overlay_simulator_integrator_method(time, etd);
  REQUIRE(time.method() == pfc::sim::steppers::RKIntegratorMethod::ETD1);

  const nlohmann::json mixed = {{"restart_from", "/tmp/ckpt/step_1"},
                                {"simulator", {{"increment", 4}}}};
  REQUIRE_THROWS_AS(pfc::ui::reject_mixed_restart_keys(mixed),
                    std::invalid_argument);
}
