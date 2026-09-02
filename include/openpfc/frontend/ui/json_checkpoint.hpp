// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file json_checkpoint.hpp
 * @brief JSON `restart_from` / `checkpoint.*` for 0.2 sessions (M11/M12).
 *
 * `restart_from` is exclusive of leftover `simulator.increment` /
 * `simulator.result_counter` keys. Sessions restore through
 * `CheckpointService` (`restore_from_config` on `SimulationState` + `Time`).
 */

#include <stdexcept>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/kernel/simulation/checkpoint_service.hpp>

namespace pfc::ui {

inline void reject_mixed_restart_keys(const nlohmann::json &settings) {
  const auto cfg = pfc::sim::checkpoint_config_from_json(settings);
  if (cfg.restart_from.empty()) {
    return;
  }
  if (!settings.contains("simulator") || !settings["simulator"].is_object()) {
    return;
  }
  const auto &j = settings["simulator"];
  if (j.contains("increment") || j.contains("result_counter")) {
    throw std::invalid_argument(
        "restart_from cannot be combined with simulator.increment or "
        "simulator.result_counter; the checkpoint bundle restores Time and "
        "the result counter");
  }
}

[[nodiscard]] inline pfc::sim::CheckpointService
make_checkpoint_service(const nlohmann::json &settings, MPI_Comm comm) {
  reject_mixed_restart_keys(settings);
  return pfc::sim::CheckpointService(pfc::sim::checkpoint_config_from_json(settings),
                                     comm);
}

} // namespace pfc::ui
