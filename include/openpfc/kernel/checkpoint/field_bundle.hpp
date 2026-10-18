// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file field_bundle.hpp
 * @brief Save and restore a named set of distributed float64 fields as one
 *        checkpoint bundle (MPI-collective).
 *
 * @details
 * `publish_field_bundle` packs the owned cells of every field, fills
 * `CheckpointMetadata` (domain from the first field, decomposition, caller
 * time/increment/method identity and boundary records), writes one brick per
 * field and an optional caller-owned JSON sidecar, and publishes everything
 * through `publish_checkpoint_directory`. `load_field_bundle` reads the
 * metadata on rank 0, broadcasts it, requires the same field names in the same
 * order and the same global grid, and reads each brick into the owned cells
 * of the caller's fields. Any decomposition may restore any other: bricks are
 * global and x-fastest. Halos are neither stored nor refreshed.
 *
 * Fields must share one domain and one decomposition. Host and device fields
 * are supported, with or without storage halos. Device fields go through
 * their existing host mirror; only owned cells enter the bundle.
 *
 * `pfc::sim::CheckpointService` builds on these functions for
 * `SimulationState` sessions.
 *
 * @see publish.hpp, checkpoint_metadata.hpp, brick_io.hpp
 * @see docs/development/checkpoint_publish.md
 */

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/kernel/checkpoint/brick_io.hpp>
#include <openpfc/kernel/checkpoint/checkpoint_metadata.hpp>
#include <openpfc/kernel/checkpoint/publish.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
#include <openpfc/kernel/simulation/binary_reader.hpp>

namespace pfc::checkpoint {

/// Field id in the bundle (`fields/<id>.bin`) and the field it stores.
template <class MemorySpace = pfc::HostSpace>
using NamedField = std::pair<std::string, pfc::data::Field<double, MemorySpace> *>;

/// Accepted state written into `metadata.json` next to the fields.
struct FieldBundleState {
  double accepted_time{0.0};
  int accepted_increment{0};
  int result_counter{0};
  std::string method_identity{};
  /// Boundary records; null omits the key, as in older bundles.
  nlohmann::json boundary_conditions = nullptr;
};

/// Optional caller-owned JSON written by rank 0 as `<bundle>/<name>`.
struct FieldBundleSidecar {
  std::string name{};
  nlohmann::json content = nullptr;
};

/**
 * @brief Copy the owned cells of @p f (x-fastest, no halo) into a transport
 *        buffer. Device fields go through the host mirror.
 */
template <typename T, typename MemorySpace>
[[nodiscard]] std::vector<double>
pack_owned_real(pfc::data::Field<T, MemorySpace> &f) {
  static_assert(std::is_same_v<T, double>, "pack_owned_real is float64 only");
  const auto sz = f.local_size();
  std::vector<double> out(static_cast<std::size_t>(sz[0]) *
                          static_cast<std::size_t>(sz[1]) *
                          static_cast<std::size_t>(sz[2]));
  f.with_host_read([&](const T *data, std::size_t) {
    std::size_t n = 0;
    for (int k = 0; k < sz[2]; ++k) {
      for (int j = 0; j < sz[1]; ++j) {
        for (int i = 0; i < sz[0]; ++i) {
          out[n++] = data[f.idx(i, j, k)];
        }
      }
    }
  });
  return out;
}

/** @brief Inverse of @ref pack_owned_real. */
template <typename T, typename MemorySpace>
inline void unpack_owned_real(pfc::data::Field<T, MemorySpace> &f,
                              const std::vector<double> &in) {
  static_assert(std::is_same_v<T, double>, "unpack_owned_real is float64 only");
  const auto sz = f.local_size();
  const auto owned = static_cast<std::size_t>(sz[0]) *
                     static_cast<std::size_t>(sz[1]) *
                     static_cast<std::size_t>(sz[2]);
  if (in.size() != owned) {
    throw std::invalid_argument("unpack_owned_real: owned field size mismatch");
  }
  f.with_host_view([&](T *data, std::size_t) {
    std::size_t n = 0;
    for (int k = 0; k < sz[2]; ++k) {
      for (int j = 0; j < sz[1]; ++j) {
        for (int i = 0; i < sz[0]; ++i) {
          data[f.idx(i, j, k)] = in[n++];
        }
      }
    }
  });
  f.sync_to_device();
}

/** @brief Write the owned cells of @p f as brick @p path (collective). */
template <class MemorySpace>
inline void write_field_brick(const std::filesystem::path &path, MPI_Comm comm,
                              pfc::data::Field<double, MemorySpace> &f) {
  const auto brick = pack_owned_real(f);
  const auto loc = f.local_size();
  const auto off = f.lower_global();
  const auto gsz = f.global_size();
  write_real_brick_mpi(path.string(), comm, {gsz[0], gsz[1], gsz[2]},
                       {loc[0], loc[1], loc[2]}, {off[0], off[1], off[2]}, brick);
}

/** @brief Read brick @p path into the owned cells of @p f (collective). */
template <class MemorySpace>
inline void read_field_brick(const std::filesystem::path &path, MPI_Comm comm,
                             pfc::data::Field<double, MemorySpace> &f) {
  const auto loc = f.local_size();
  const auto off = f.lower_global();
  const auto gsz = f.global_size();
  std::vector<double> brick(static_cast<std::size_t>(loc[0]) *
                            static_cast<std::size_t>(loc[1]) *
                            static_cast<std::size_t>(loc[2]));
  pfc::BinaryReader reader(comm);
  reader.set_domain({gsz[0], gsz[1], gsz[2]}, {loc[0], loc[1], loc[2]},
                    {off[0], off[1], off[2]});
  reader.read(path.string(), brick);
  unpack_owned_real(f, brick);
}

/**
 * @brief Read `<dir>/metadata.json` on rank 0 and broadcast it (collective).
 * @throws std::invalid_argument on every rank when the file is missing or
 *         unreadable, or when `from_json` rejects it.
 */
[[nodiscard]] inline CheckpointMetadata
read_checkpoint_metadata(const std::filesystem::path &dir, MPI_Comm comm) {
  int rank = 0;
  pfc::mpi::throw_on_mpi_error(MPI_Comm_rank(comm, &rank), "MPI_Comm_rank");
  std::string dump;
  int have = 1;
  if (rank == 0) {
    try {
      std::ifstream in(dir / "metadata.json");
      if (!in) {
        have = 0;
      } else {
        nlohmann::json j;
        in >> j;
        dump = j.dump();
      }
    } catch (...) {
      have = 0;
    }
  }
  pfc::mpi::throw_on_mpi_error(MPI_Bcast(&have, 1, MPI_INT, 0, comm),
                               "MPI_Bcast metadata present");
  if (have == 0) {
    throw std::invalid_argument("checkpoint load: missing metadata.json in " +
                                dir.string());
  }
  int n = static_cast<int>(dump.size());
  pfc::mpi::throw_on_mpi_error(MPI_Bcast(&n, 1, MPI_INT, 0, comm),
                               "MPI_Bcast metadata size");
  std::string buf(static_cast<std::size_t>(n), '\0');
  if (rank == 0) {
    buf = dump;
  }
  pfc::mpi::throw_on_mpi_error(MPI_Bcast(buf.data(), n, MPI_CHAR, 0, comm),
                               "MPI_Bcast metadata");
  return from_json(nlohmann::json::parse(buf));
}

/**
 * @brief Atomically publish @p fields at @p dir (MPI-collective).
 *
 * @return the `publish_checkpoint_directory` outcome (identical on every
 *         rank). Invalid input (no fields, fields on different grids or
 *         decompositions, an empty or reserved sidecar name) throws
 *         `std::invalid_argument` before anything is written.
 */
template <class MemorySpace = pfc::HostSpace>
[[nodiscard]] PublishOutcome
publish_field_bundle(const std::filesystem::path &dir,
                     const std::vector<NamedField<MemorySpace>> &fields,
                     const FieldBundleState &state, MPI_Comm comm,
                     const FieldBundleSidecar &sidecar = {}) {
  if (fields.empty()) {
    throw std::invalid_argument("publish_field_bundle: no fields");
  }
  if (!sidecar.content.is_null() &&
      (sidecar.name.empty() || sidecar.name == "metadata.json" ||
       sidecar.name == "fields" ||
       std::filesystem::path(sidecar.name).has_parent_path())) {
    throw std::invalid_argument("publish_field_bundle: invalid sidecar name '" +
                                sidecar.name + "'");
  }
  const auto &f0 = *fields.front().second;
  const auto g = f0.global_size();
  const auto n = f0.local_size();
  const auto lo = f0.lower_global();
  int size = 1;
  int rank = 0;
  pfc::mpi::throw_on_mpi_error(MPI_Comm_size(comm, &size), "MPI_Comm_size");
  pfc::mpi::throw_on_mpi_error(MPI_Comm_rank(comm, &rank), "MPI_Comm_rank");

  CheckpointMetadata meta;
  meta.accepted_time = state.accepted_time;
  meta.accepted_increment = state.accepted_increment;
  meta.result_counter = state.result_counter;
  meta.domain.global_dimensions = {g[0], g[1], g[2]};
  meta.domain.physical_origin = {f0.origin()[0], f0.origin()[1], f0.origin()[2]};
  meta.domain.grid_spacing = {f0.spacing()[0], f0.spacing()[1], f0.spacing()[2]};
  meta.decomposition = DecompositionLayout{.mpi_size = size,
                                           .local_size = {n[0], n[1], n[2]},
                                           .local_offset = {lo[0], lo[1], lo[2]}};
  meta.method_identity = state.method_identity;
  meta.boundary_conditions = state.boundary_conditions;
  for (const auto &[name, f] : fields) {
    if (f == nullptr || f->global_size() != g || f->local_size() != n ||
        f->lower_global() != lo || f->origin() != f0.origin() ||
        f->spacing() != f0.spacing()) {
      throw std::invalid_argument(
          "publish_field_bundle: fields differ in grid or layout: " + name);
    }
    meta.fields.push_back(name);
  }

  return publish_checkpoint_directory(
      dir, meta, comm, [&](const std::filesystem::path &fields_dir) {
        for (const auto &[name, f] : fields) {
          write_field_brick(fields_dir / (name + ".bin"), comm, *f);
        }
        if (rank == 0 && !sidecar.content.is_null()) {
          std::ofstream out(fields_dir.parent_path() / sidecar.name);
          out << sidecar.content.dump(2) << "\n";
          if (!out) {
            throw std::runtime_error("could not write sidecar " + sidecar.name);
          }
        }
      });
}

/**
 * @brief Restore exactly the named @p fields from @p dir (MPI-collective).
 *
 * The bundle must list the same field names in the same order, and its global
 * grid (size, origin, spacing) must equal that of every field. The current
 * decomposition may differ from the saved one.
 *
 * @return the bundle metadata (time, increment, method identity, boundary
 *         records). A sidecar, if any, is not read.
 * @throws std::invalid_argument on a name or grid mismatch, on every rank.
 */
template <class MemorySpace = pfc::HostSpace>
[[nodiscard]] CheckpointMetadata
load_field_bundle(const std::filesystem::path &dir,
                  const std::vector<NamedField<MemorySpace>> &fields,
                  MPI_Comm comm) {
  const auto meta = read_checkpoint_metadata(dir, comm);
  std::vector<std::string> names;
  for (const auto &kv : fields) {
    names.push_back(kv.first);
  }
  if (meta.fields != names) {
    throw std::invalid_argument(
        "load_field_bundle: bundle fields differ from the requested fields");
  }
  for (const auto &[name, f] : fields) {
    const auto g = f->global_size();
    for (int a = 0; a < 3; ++a) {
      if (meta.domain.global_dimensions[a] != g[a] ||
          meta.domain.grid_spacing[a] != f->spacing()[a] ||
          meta.domain.physical_origin[a] != f->origin()[a]) {
        throw std::invalid_argument(
            "load_field_bundle: bundle grid differs from field " + name);
      }
    }
  }
  for (const auto &[name, f] : fields) {
    read_field_brick(dir / "fields" / (name + ".bin"), comm, *f);
  }
  return meta;
}

} // namespace pfc::checkpoint
