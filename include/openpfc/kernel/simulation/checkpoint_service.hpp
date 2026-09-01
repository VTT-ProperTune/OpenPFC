// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file checkpoint_service.hpp
 * @brief Framework owner of checkpoint save/load (M11).
 *
 * JSON keys: `checkpoint.every`, `checkpoint.directory`, `restart_from`.
 * Bundles are `<directory>/step_<increment>/` with `metadata.json` and
 * `fields/<name>.bin` (collective MPI-IO bricks). Interrupted publishes
 * leave only a `.publishing` staging dir, never a loadable `final_dir`.
 */

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

template <typename T>
[[nodiscard]] RealField pack_owned_real(const pfc::data::Field<T> &f) {
  static_assert(std::is_same_v<T, double>, "pack_owned_real is float64 only");
  const auto sz = f.local_size();
  RealField out(static_cast<std::size_t>(sz[0]) * static_cast<std::size_t>(sz[1]) *
                static_cast<std::size_t>(sz[2]));
  std::size_t n = 0;
  for (int k = 0; k < sz[2]; ++k) {
    for (int j = 0; j < sz[1]; ++j) {
      for (int i = 0; i < sz[0]; ++i) {
        out[n++] = f(i, j, k);
      }
    }
  }
  return out;
}

template <typename T>
inline void unpack_owned_real(pfc::data::Field<T> &f, const RealField &in) {
  const auto sz = f.local_size();
  std::size_t n = 0;
  for (int k = 0; k < sz[2]; ++k) {
    for (int j = 0; j < sz[1]; ++j) {
      for (int i = 0; i < sz[0]; ++i) {
        f(i, j, k) = in[n++];
      }
    }
  }
}

class CheckpointService {
public:
  CheckpointService(CheckpointConfig cfg, MPI_Comm comm = MPI_COMM_WORLD)
      : m_cfg(std::move(cfg)), m_comm(comm) {}

  [[nodiscard]] const CheckpointConfig &config() const noexcept { return m_cfg; }
  [[nodiscard]] int result_counter() const noexcept { return m_result_counter; }
  void set_result_counter(int c) noexcept { m_result_counter = c; }

  [[nodiscard]] std::filesystem::path step_dir(int increment) const {
    std::ostringstream oss;
    oss << "step_" << increment;
    return m_cfg.directory / oss.str();
  }

  void save(const SimulationState &state, const Time &time) {
    if (m_cfg.directory.empty()) {
      throw std::invalid_argument("CheckpointService::save: directory is empty");
    }
    std::vector<std::string> names;
    for (const auto &name : state.field_names()) {
      if (state.has_typed_field<double>(name)) {
        names.push_back(name);
      }
    }
    if (names.empty()) {
      throw std::invalid_argument("CheckpointService::save: no fields");
    }
    const auto &field0 = state.get_field<double>(names.front());
    const auto &dom = field0.domain();
    const auto sz = field0.local_size();
    const auto lo = field0.box().low;

    int nproc = 1;
    int rank = 0;
    pfc::mpi::throw_on_mpi_error(MPI_Comm_size(m_comm, &nproc), "MPI_Comm_size");
    pfc::mpi::throw_on_mpi_error(MPI_Comm_rank(m_comm, &rank), "MPI_Comm_rank");

    checkpoint::CheckpointMetadata meta;
    meta.accepted_time = time.get_current();
    meta.accepted_increment = time.get_increment();
    meta.result_counter = m_result_counter;
    meta.domain = domain_params_from(dom);
    meta.method_identity = pfc::sim::steppers::to_string(time.method());
    meta.fields = names;
    checkpoint::DecompositionMeta dm;
    dm.mpi_size = nproc;
    dm.local_size = {sz[0], sz[1], sz[2]};
    dm.local_offset = {lo[0], lo[1], lo[2]};
    meta.decomposition = dm;

    const auto final_dir = step_dir(time.get_increment());
    const auto staging = std::filesystem::path(final_dir.string() + ".publishing");

    int exists_flag = 0;
    if (rank == 0) {
      exists_flag = std::filesystem::exists(final_dir) ? 1 : 0;
    }
    pfc::mpi::throw_on_mpi_error(MPI_Bcast(&exists_flag, 1, MPI_INT, 0, m_comm),
                                 "MPI_Bcast ckpt exists");
    if (exists_flag != 0) {
      throw std::runtime_error("checkpoint already exists: " + final_dir.string());
    }

    if (rank == 0) {
      std::error_code ec;
      std::filesystem::remove_all(staging, ec);
      std::filesystem::create_directories(staging / "fields");
    }
    pfc::mpi::throw_on_mpi_error(MPI_Barrier(m_comm), "MPI_Barrier ckpt stage");

    std::exception_ptr eptr;
    try {
      for (const auto &name : names) {
        const auto &f = state.get_field<double>(name);
        const auto brick = pack_owned_real(f);
        const auto loc = f.local_size();
        const auto off = f.box().low;
        const auto gsz = f.global_size();
        const auto path = (staging / "fields" / (name + ".bin")).string();
        checkpoint::write_real_brick_mpi(path, m_comm, {gsz[0], gsz[1], gsz[2]},
                                         {loc[0], loc[1], loc[2]},
                                         {off[0], off[1], off[2]}, brick);
      }
      if (rank == 0) {
        std::ofstream out(staging / "metadata.json");
        if (!out) {
          throw std::runtime_error("failed to write metadata.json");
        }
        out << checkpoint::to_json(meta).dump(2) << '\n';
      }
    } catch (...) {
      eptr = std::current_exception();
    }
    int local_ok = eptr ? 0 : 1;
    int global_ok = 0;
    pfc::mpi::throw_on_mpi_error(
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, m_comm),
        "MPI_Allreduce ckpt write");
    if (global_ok == 0) {
      if (rank == 0) {
        std::error_code ec;
        std::filesystem::remove_all(staging, ec);
        std::filesystem::remove_all(final_dir, ec);
      }
      pfc::mpi::throw_on_mpi_error(MPI_Barrier(m_comm), "MPI_Barrier ckpt fail");
      if (eptr) {
        std::rethrow_exception(eptr);
      }
      throw std::runtime_error("checkpoint save failed on another rank");
    }
    if (rank == 0) {
      std::filesystem::rename(staging, final_dir);
    }
    pfc::mpi::throw_on_mpi_error(MPI_Barrier(m_comm), "MPI_Barrier ckpt rename");
  }

  [[nodiscard]] checkpoint::CheckpointMetadata
  read_metadata(const std::filesystem::path &dir) const {
    int rank = 0;
    pfc::mpi::throw_on_mpi_error(MPI_Comm_rank(m_comm, &rank), "MPI_Comm_rank");

    nlohmann::json j;
    std::string dump;
    int have = 1;
    if (rank == 0) {
      try {
        std::ifstream in(dir / "metadata.json");
        if (!in) {
          have = 0;
        } else {
          in >> j;
          dump = j.dump();
        }
      } catch (...) {
        have = 0;
      }
    }
    pfc::mpi::throw_on_mpi_error(MPI_Bcast(&have, 1, MPI_INT, 0, m_comm),
                                 "MPI_Bcast metadata present");
    if (have == 0) {
      throw std::invalid_argument("checkpoint load: missing metadata.json in " +
                                  dir.string());
    }
    int n = static_cast<int>(dump.size());
    pfc::mpi::throw_on_mpi_error(MPI_Bcast(&n, 1, MPI_INT, 0, m_comm),
                                 "MPI_Bcast metadata size");
    std::string buf(static_cast<std::size_t>(n), '\0');
    if (rank == 0) {
      buf = dump;
    }
    pfc::mpi::throw_on_mpi_error(MPI_Bcast(buf.data(), n, MPI_CHAR, 0, m_comm),
                                 "MPI_Bcast metadata");
    return checkpoint::from_json(nlohmann::json::parse(buf));
  }

  void load(SimulationState &state, Time &time, const std::filesystem::path &dir) {
    const auto meta = read_metadata(dir);

    std::vector<std::string> names = meta.fields;
    if (names.empty()) {
      for (const auto &name : state.field_names()) {
        if (state.has_typed_field<double>(name)) {
          names.push_back(name);
        }
      }
    }
    if (names.empty()) {
      throw std::invalid_argument("checkpoint load: no field names");
    }
    const auto &field0 = state.get_field<double>(names.front());
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

    for (const auto &name : names) {
      auto &f = state.get_field<double>(name);
      const auto loc = f.local_size();
      const auto off = f.box().low;
      const auto gsz = f.global_size();
      RealField brick(static_cast<std::size_t>(loc[0]) *
                      static_cast<std::size_t>(loc[1]) *
                      static_cast<std::size_t>(loc[2]));
      BinaryReader reader(m_comm);
      reader.set_domain({gsz[0], gsz[1], gsz[2]}, {loc[0], loc[1], loc[2]},
                        {off[0], off[1], off[2]});
      const auto path = (dir / "fields" / (name + ".bin")).string();
      reader.read(path, brick);
      unpack_owned_real(f, brick);
    }
  }

  bool maybe_save(const SimulationState &state, const Time &time) {
    if (m_cfg.every <= 0 || m_cfg.directory.empty()) {
      return false;
    }
    const int inc = time.get_increment();
    if (inc == 0 || (inc % m_cfg.every) != 0) {
      return false;
    }
    save(state, time);
    return true;
  }

  void restore_from_config(SimulationState &state, Time &time) {
    if (m_cfg.restart_from.empty()) {
      return;
    }
    load(state, time, m_cfg.restart_from);
  }

private:
  CheckpointConfig m_cfg;
  MPI_Comm m_comm{MPI_COMM_WORLD};
  int m_result_counter{0};
};

} // namespace pfc::sim
