// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulation_wiring_simulator_section.hpp
 * @brief Optional top-level JSON `"simulator"` keys (`result_counter`, `increment`)
 */

#ifndef PFC_UI_SIMULATION_WIRING_SIMULATOR_SECTION_HPP
#define PFC_UI_SIMULATION_WIRING_SIMULATOR_SECTION_HPP

#include <stdexcept>

#include <nlohmann/json.hpp>
#include <openpfc/kernel/simulation/simulator.hpp>
#include <openpfc/kernel/simulation/time.hpp>

namespace pfc::ui {

/**
 * @brief Apply optional top-level `"simulator"` object (`result_counter`,
 * `increment`)
 *
 * `result_counter` in JSON is treated as the last completed index; the simulator
 * counter is set to that value plus one (same as previous `App` behavior).
 */
inline void apply_simulator_section_from_json(Simulator &sim, Time &time,
                                              const nlohmann::json &settings) {
  if (!settings.contains("simulator")) {
    return;
  }
  const nlohmann::json &j = settings["simulator"];
  if (j.contains("result_counter")) {
    if (!j["result_counter"].is_number_integer()) {
      throw std::invalid_argument(
          "Invalid JSON input: missing or invalid 'result_counter' field.");
    }
    const int result_counter = static_cast<int>(j["result_counter"]) + 1;
    pfc::set_result_counter(sim, result_counter);
  }
  if (j.contains("increment")) {
    if (!j["increment"].is_number_integer()) {
      throw std::invalid_argument(
          "Invalid JSON input: missing or invalid 'increment' field.");
    }
    const int increment = static_cast<int>(j["increment"]);
    time.set_increment(increment);
  }
}

} // namespace pfc::ui

#endif // PFC_UI_SIMULATION_WIRING_SIMULATOR_SECTION_HPP
