// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file json_checkpoint.hpp
 * @brief JSON `restart_from` / `checkpoint.*` for App and 0.2 sessions (M11).
 *
 * `restart_from` is exclusive of Gen-1 `simulator.increment` /
 * `simulator.result_counter`. 0.2 sessions (`SimulationState`) use
 * `CheckpointService` directly. Gen-1 `Simulator` restores real Model
 * fields through `BinaryReader` using the FFT inbox layout.
 */

#include <filesystem>
#include <stdexcept>
#include <string>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/simulation/binary_reader.hpp>
#include <openpfc/kernel/simulation/checkpoint_service.hpp>
#include <openpfc/kernel/simulation/model.hpp>
#include <openpfc/kernel/simulation/simulator.hpp>
#include <openpfc/kernel/simulation/time.hpp>

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

inline void restore_gen1_from_checkpoint(Simulator &sim, Time &time,
                                         const std::filesystem::path &dir) {
  pfc::sim::CheckpointService svc({.every = 0, .restart_from = dir}, sim.mpi_comm());
  const auto meta = svc.read_metadata(dir);
  pfc::Model &model = sim.get_model();
  const auto &world = pfc::get_world(model);
  const auto world_size = pfc::world::get_size(world);
  const auto origin = pfc::world::get_origin(world);
  const auto spacing = pfc::world::get_spacing(world);
  pfc::sim::require_checkpoint_identity(
      meta, pfc::checkpoint::CheckpointMetadata{
                .format_version = pfc::checkpoint::kCheckpointFormatVersion,
                .domain = {.global_dimensions = {world_size[0], world_size[1],
                                                 world_size[2]},
                           .physical_origin = {origin[0], origin[1], origin[2]},
                           .grid_spacing = {spacing[0], spacing[1], spacing[2]}},
                .method_identity = pfc::sim::steppers::to_string(time.method()),
            });

  time.set_increment(meta.accepted_increment);
  if (!meta.method_identity.empty()) {
    auto parsed = pfc::sim::steppers::resolve_method_id(meta.method_identity);
    if (!parsed) {
      throw std::invalid_argument("checkpoint load: unknown method_identity '" +
                                  meta.method_identity + "'");
    }
    time.set_method(*parsed);
  }
  pfc::set_result_counter(sim, meta.result_counter);

  const auto inbox = pfc::fft::get_inbox(pfc::get_fft(model));
  pfc::BinaryReader reader(sim.mpi_comm());
  reader.set_domain(world_size, inbox.size, inbox.low);
  for (const auto &name : meta.fields) {
    if (!model.has_real_field(name)) {
      continue;
    }
    const auto path = (dir / "fields" / (name + ".bin")).string();
    reader.read(path, model.get_real_field(name));
  }
}

} // namespace pfc::ui
