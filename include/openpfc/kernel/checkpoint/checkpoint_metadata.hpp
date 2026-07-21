// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file checkpoint_metadata.hpp
 * @brief Versioned metadata for filesystem checkpoint publication
 *
 * @details
 * `CheckpointMetadata` records the irreducible restart identity for an
 * accepted solution state: format version, accepted simulation time and
 * increment, domain parameters, optional MPI decomposition descriptors, and
 * integrator method identity.
 *
 * Callers must fill `accepted_time` and `accepted_increment` from
 * driver-owned `pfc::sim::Time` (`get_current()` / `get_increment()`). This
 * header does not construct or advance `Time`.
 *
 * @see publish.hpp for atomic directory publication
 * @see docs/development/checkpoint_publish.md
 */

#ifndef OPENPFC_KERNEL_CHECKPOINT_CHECKPOINT_METADATA_HPP
#define OPENPFC_KERNEL_CHECKPOINT_CHECKPOINT_METADATA_HPP

#include <array>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

namespace pfc::checkpoint {

/// On-disk metadata schema version for published checkpoint bundles.
inline constexpr int kCheckpointFormatVersion = 1;

/**
 * @brief Global domain geometry recorded in checkpoint metadata.
 */
struct DomainParams {
  std::array<int, 3> global_dimensions{};
  std::array<double, 3> physical_origin{};
  std::array<double, 3> grid_spacing{};
};

/**
 * @brief Optional MPI decomposition layout for a sibling restore leaf.
 */
struct DecompositionMeta {
  int mpi_size{1};
  std::array<int, 3> local_size{};
  std::array<int, 3> local_offset{};
};

/**
 * @brief Versioned checkpoint sidecar: accepted time/increment + domain.
 *
 * @note `accepted_time` / `accepted_increment` must come from
 *       `pfc::sim::Time::get_current()` / `get_increment()` (caller fills).
 */
struct CheckpointMetadata {
  int format_version{kCheckpointFormatVersion};
  double accepted_time{0.0};
  int accepted_increment{0};
  DomainParams domain{};
  std::optional<DecompositionMeta> decomposition{};
  std::string method_identity{};
};

/**
 * @brief Serialize @p meta to JSON for `metadata.json` in a checkpoint bundle.
 *
 * Omits `decomposition` when nullopt; always emits `method_identity`.
 */
[[nodiscard]] inline nlohmann::json to_json(const CheckpointMetadata &meta) {
  nlohmann::json j;
  j["format_version"] = meta.format_version;
  j["accepted_time"] = meta.accepted_time;
  j["accepted_increment"] = meta.accepted_increment;
  j["domain"] = {
      {"global_dimensions", meta.domain.global_dimensions},
      {"physical_origin", meta.domain.physical_origin},
      {"grid_spacing", meta.domain.grid_spacing},
  };
  if (meta.decomposition.has_value()) {
    const auto &d = *meta.decomposition;
    j["decomposition"] = {
        {"mpi_size", d.mpi_size},
        {"local_size", d.local_size},
        {"local_offset", d.local_offset},
    };
  }
  j["method_identity"] = meta.method_identity;
  return j;
}

} // namespace pfc::checkpoint

#endif
