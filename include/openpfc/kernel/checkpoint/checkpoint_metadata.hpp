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
#include <stdexcept>
#include <string>
#include <vector>

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
 * @brief Optional MPI decomposition layout recorded in checkpoint JSON.
 *
 * Distinct from `payloads.hpp`'s `DecompositionMeta` (in-memory field
 * payload). Do not include both headers in one translation unit.
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
  int result_counter{0};
  DomainParams domain{};
  std::optional<DecompositionMeta> decomposition{};
  std::string method_identity{};
  std::vector<std::string> fields{};
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
  j["result_counter"] = meta.result_counter;
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
  j["fields"] = meta.fields;
  return j;
}

/**
 * @brief Parse @p j into `CheckpointMetadata`.
 *
 * Requires `format_version` equal to @ref kCheckpointFormatVersion.
 * Missing or mistyped required keys throw `std::invalid_argument` naming
 * the field.
 */
[[nodiscard]] inline CheckpointMetadata from_json(const nlohmann::json &j) {
  auto require_key = [&](const char *key) -> const nlohmann::json & {
    if (!j.contains(key)) {
      throw std::invalid_argument(
          std::string("CheckpointMetadata: missing required key '") + key + "'");
    }
    return j.at(key);
  };

  CheckpointMetadata meta;
  const auto &ver = require_key("format_version");
  if (!ver.is_number_integer()) {
    throw std::invalid_argument(
        "CheckpointMetadata: format_version must be an integer");
  }
  meta.format_version = ver.get<int>();
  if (meta.format_version != kCheckpointFormatVersion) {
    throw std::invalid_argument(
        "CheckpointMetadata schema version mismatch: file has " +
        std::to_string(meta.format_version) + ", expected " +
        std::to_string(kCheckpointFormatVersion));
  }

  meta.accepted_time = require_key("accepted_time").get<double>();
  meta.accepted_increment = require_key("accepted_increment").get<int>();
  if (j.contains("result_counter")) {
    meta.result_counter = j["result_counter"].get<int>();
  }
  const auto &dom = require_key("domain");
  meta.domain.global_dimensions =
      dom.at("global_dimensions").get<std::array<int, 3>>();
  meta.domain.physical_origin =
      dom.at("physical_origin").get<std::array<double, 3>>();
  meta.domain.grid_spacing = dom.at("grid_spacing").get<std::array<double, 3>>();

  if (j.contains("decomposition") && !j["decomposition"].is_null()) {
    const auto &d = j["decomposition"];
    DecompositionMeta dm;
    dm.mpi_size = d.at("mpi_size").get<int>();
    dm.local_size = d.at("local_size").get<std::array<int, 3>>();
    dm.local_offset = d.at("local_offset").get<std::array<int, 3>>();
    meta.decomposition = dm;
  }
  meta.method_identity = require_key("method_identity").get<std::string>();
  if (j.contains("fields") && j["fields"].is_array()) {
    meta.fields = j["fields"].get<std::vector<std::string>>();
  }
  return meta;
}

} // namespace pfc::checkpoint

#endif
