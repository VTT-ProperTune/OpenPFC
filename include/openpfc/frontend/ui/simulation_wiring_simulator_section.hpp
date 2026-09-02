// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulation_wiring_simulator_section.hpp
 * @brief Optional top-level JSON `"simulator"` keys (`increment`, integrator)
 */

#ifndef PFC_UI_SIMULATION_WIRING_SIMULATOR_SECTION_HPP
#define PFC_UI_SIMULATION_WIRING_SIMULATOR_SECTION_HPP

#include <stdexcept>

#include <nlohmann/json.hpp>
#include <openpfc/frontend/ui/from_json_integrator_method.hpp>
#include <openpfc/frontend/ui/json_checkpoint.hpp>
#include <openpfc/kernel/simulation/time.hpp>

namespace pfc::ui {

/**
 * @brief Overlay optional `"simulator"` object onto @p time
 *
 * `increment` sets `Time::set_increment`. `integrator.method` overlays
 * `Time::method()` after `from_json<Time>`. `restart_from` cannot be combined
 * with `increment` / `result_counter` (checkpoint restore is
 * `CheckpointService::restore_from_config`). `result_counter` is ignored:
 * dump indices come from `Time` / the checkpoint bundle.
 */
inline void apply_simulator_section_from_json(Time &time,
                                              const nlohmann::json &settings) {
  reject_mixed_restart_keys(settings);
  if (!settings.contains("simulator") || !settings["simulator"].is_object()) {
    return;
  }
  const nlohmann::json &j = settings["simulator"];
  if (j.contains("increment")) {
    if (!j["increment"].is_number_integer()) {
      throw std::invalid_argument(
          "Invalid JSON input: missing or invalid 'increment' field.");
    }
    time.set_increment(static_cast<int>(j["increment"]));
  }
  if (j.contains("integrator") && j["integrator"].is_object() &&
      j["integrator"].contains("method")) {
    time.set_method(from_json<pfc::sim::steppers::RKIntegratorMethod>(
        j["integrator"]["method"]));
  }
}

} // namespace pfc::ui

#endif // PFC_UI_SIMULATION_WIRING_SIMULATOR_SECTION_HPP
