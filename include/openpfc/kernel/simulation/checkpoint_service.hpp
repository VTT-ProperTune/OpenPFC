// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file checkpoint_service.hpp
 * @brief Framework owner of checkpoint save/load (M11).
 *
 * JSON keys: `checkpoint.every`, `checkpoint.directory`, `restart_from`.
 * Bundles are `<directory>/step_<increment>/` with `metadata.json` and
 * `fields/<name>.bin` (collective MPI-IO bricks via `brick_io.hpp`, published
 * through the one `publish_checkpoint_directory` protocol). Interrupted
 * publishes leave only a `.publishing` staging dir, never a loadable
 * `final_dir`. Every method is templated on the fields' `MemorySpace` so GPU
 * sessions checkpoint through the host mirror (`Field::with_host_view`).
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/kernel/checkpoint/brick_io.hpp>
#include <openpfc/kernel/checkpoint/checkpoint_metadata.hpp>
#include <openpfc/kernel/checkpoint/field_bundle.hpp>
#include <openpfc/kernel/checkpoint/publish.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
#include <openpfc/kernel/simulation/binary_reader.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/steppers/integrator_method.hpp>
#include <openpfc/kernel/simulation/steppers/method_composition.hpp>
#include <openpfc/kernel/simulation/time.hpp>

namespace pfc::sim {

struct CheckpointConfig {
  int every{0};
  std::filesystem::path directory{};
  std::filesystem::path restart_from{};
};

[[nodiscard]] inline CheckpointConfig
checkpoint_config_from_json(const nlohmann::json &settings) {
  CheckpointConfig cfg;
  if (settings.contains("restart_from") && settings["restart_from"].is_string()) {
    cfg.restart_from = settings["restart_from"].get<std::string>();
  }
  if (!settings.contains("checkpoint") || !settings["checkpoint"].is_object()) {
    return cfg;
  }
  const auto &j = settings["checkpoint"];
  if (j.contains("every")) {
    cfg.every = j["every"].get<int>();
  }
  if (j.contains("directory") && j["directory"].is_string()) {
    cfg.directory = j["directory"].get<std::string>();
  }
  return cfg;
}

[[nodiscard]] inline std::array<int, 3> domain_size_array(const pfc::Domain &d) {
  return d.size;
}

[[nodiscard]] inline checkpoint::DomainParams
domain_params_from(const pfc::Domain &d) {
  return checkpoint::DomainParams{
      .global_dimensions = domain_size_array(d),
      .physical_origin = {d.origin[0], d.origin[1], d.origin[2]},
      .grid_spacing = {d.spacing[0], d.spacing[1], d.spacing[2]},
  };
}

inline void require_checkpoint_identity(const checkpoint::CheckpointMetadata &file,
                                        const checkpoint::CheckpointMetadata &want) {
  if (file.format_version != want.format_version) {
    throw std::invalid_argument(
        "checkpoint identity mismatch: schema version file=" +
        std::to_string(file.format_version) +
        " expected=" + std::to_string(want.format_version));
  }
  if (file.domain.global_dimensions != want.domain.global_dimensions) {
    throw std::invalid_argument(
        "checkpoint identity mismatch: grid (global_dimensions)");
  }
  if (file.domain.grid_spacing != want.domain.grid_spacing ||
      file.domain.physical_origin != want.domain.physical_origin) {
    throw std::invalid_argument(
        "checkpoint identity mismatch: grid (origin/spacing)");
  }
  if (!want.method_identity.empty() &&
      file.method_identity != want.method_identity) {
    throw std::invalid_argument("checkpoint identity mismatch: method file='" +
                                file.method_identity + "' expected='" +
                                want.method_identity + "'");
  }
}

// Owned-cell packing lives with the bundle functions in kernel/checkpoint.
using checkpoint::pack_owned_real;
using checkpoint::unpack_owned_real;

class CheckpointService {
public:
  CheckpointService(CheckpointConfig cfg, MPI_Comm comm = MPI_COMM_WORLD)
      : m_cfg(std::move(cfg)), m_comm(comm) {}

  [[nodiscard]] const CheckpointConfig &config() const noexcept { return m_cfg; }
  [[nodiscard]] int result_counter() const noexcept { return m_result_counter; }
  void set_result_counter(int c) noexcept { m_result_counter = c; }

  /// Session-owned BC records, published atomically with fields and time.
  void set_boundary_conditions(nlohmann::json records) {
    m_boundary_conditions = std::move(records);
  }
  /// Records from the last load (null for legacy field-only bundles).
  [[nodiscard]] const nlohmann::json &boundary_conditions() const noexcept {
    return m_boundary_conditions;
  }

  [[nodiscard]] std::filesystem::path step_dir(int increment) const {
    std::ostringstream oss;
    oss << "step_" << increment;
    return m_cfg.directory / oss.str();
  }

  /**
   * @brief Publish every `double` field of @p state in @p MemorySpace.
   *
   * Collective over the service communicator. Device fields are read through
   * their host mirror; the residency bracket is owned here, not by the caller.
   */
  template <class MemorySpace = pfc::HostSpace>
  void save(SimulationState &state, const Time &time) {
    if (m_cfg.directory.empty()) {
      throw std::invalid_argument("CheckpointService::save: directory is empty");
    }
    std::vector<std::string> names;
    for (const auto &name : state.field_names()) {
      if (state.has_typed_field<double, MemorySpace>(name)) {
        names.push_back(name);
      }
    }
    if (names.empty()) {
      throw std::invalid_argument("CheckpointService::save: no fields");
    }
    std::vector<checkpoint::NamedField<MemorySpace>> fields;
    for (const auto &name : names) {
      fields.emplace_back(name, &state.get_field<double, MemorySpace>(name));
    }
    checkpoint::FieldBundleState bundle;
    bundle.accepted_time = time.get_current();
    bundle.accepted_increment = time.get_increment();
    bundle.result_counter = m_result_counter;
    bundle.method_identity = pfc::sim::steppers::to_string(time.method());
    bundle.boundary_conditions = m_boundary_conditions;
    const auto outcome = checkpoint::publish_field_bundle(
        step_dir(time.get_increment()), fields, bundle, m_comm);
    if (!outcome.ok) {
      throw std::runtime_error("CheckpointService::save: " + outcome.message);
    }
  }

  [[nodiscard]] checkpoint::CheckpointMetadata
  read_metadata(const std::filesystem::path &dir) const {
    return checkpoint::read_checkpoint_metadata(dir, m_comm);
  }

  template <class MemorySpace = pfc::HostSpace>
  void load(SimulationState &state, Time &time, const std::filesystem::path &dir) {
    const auto meta = read_metadata(dir);

    std::vector<std::string> names = meta.fields;
    if (names.empty()) {
      for (const auto &name : state.field_names()) {
        if (state.has_typed_field<double, MemorySpace>(name)) {
          names.push_back(name);
        }
      }
    }
    if (names.empty()) {
      throw std::invalid_argument("checkpoint load: no field names");
    }
    const auto &field0 = state.get_field<double, MemorySpace>(names.front());
    checkpoint::CheckpointMetadata want;
    want.format_version = checkpoint::kCheckpointFormatVersion;
    want.domain = domain_params_from(field0.domain());
    want.method_identity = pfc::sim::steppers::to_string(time.method());
    require_checkpoint_identity(meta, want);

    time.set_increment(meta.accepted_increment);
    if (std::abs(time.get_current() - meta.accepted_time) > 1e-9) {
      throw std::invalid_argument(
          "checkpoint identity mismatch: accepted_time file=" +
          std::to_string(meta.accepted_time) +
          " reconstructed=" + std::to_string(time.get_current()));
    }
    if (!meta.method_identity.empty()) {
      auto parsed = pfc::sim::steppers::resolve_method_id(meta.method_identity);
      if (!parsed) {
        throw std::invalid_argument("checkpoint load: unknown method_identity '" +
                                    meta.method_identity + "'");
      }
      time.set_method(*parsed);
    }
    m_result_counter = meta.result_counter;
    m_boundary_conditions = meta.boundary_conditions;

    for (const auto &name : names) {
      checkpoint::read_field_brick(dir / "fields" / (name + ".bin"), m_comm,
                                   state.get_field<double, MemorySpace>(name));
    }
  }

  template <class MemorySpace = pfc::HostSpace>
  bool maybe_save(SimulationState &state, const Time &time) {
    if (m_cfg.every <= 0 || m_cfg.directory.empty()) {
      return false;
    }
    const int inc = time.get_increment();
    if (inc == 0 || (inc % m_cfg.every) != 0) {
      return false;
    }
    save<MemorySpace>(state, time);
    return true;
  }

  template <class MemorySpace = pfc::HostSpace>
  void restore_from_config(SimulationState &state, Time &time) {
    if (m_cfg.restart_from.empty()) {
      return;
    }
    load<MemorySpace>(state, time, m_cfg.restart_from);
  }

private:
  CheckpointConfig m_cfg;
  MPI_Comm m_comm{MPI_COMM_WORLD};
  int m_result_counter{0};
  nlohmann::json m_boundary_conditions = nullptr;
};

} // namespace pfc::sim
