// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file from_json_field_modifiers.hpp
 * @brief JSON hooks for built-in IC/BC types
 */

#ifndef PFC_UI_FROM_JSON_FIELD_MODIFIERS_HPP
#define PFC_UI_FROM_JSON_FIELD_MODIFIERS_HPP

#include <stdexcept>
#include <string_view>

#include <openpfc/frontend/ui/from_json_fwd.hpp>
#include <openpfc/kernel/simulation/initial_conditions/constant.hpp>
#include <openpfc/kernel/simulation/initial_conditions/file_reader.hpp>
#include <openpfc/kernel/simulation/initial_conditions/random_seeds.hpp>
#include <openpfc/kernel/simulation/initial_conditions/seed_grid.hpp>
#include <openpfc/kernel/simulation/initial_conditions/single_seed.hpp>

namespace pfc::ui {
namespace detail {

inline void throw_unless_json_modifier_type(const json &j, const char *expected,
                                            std::string_view message) {
  if (!j.contains("type") || j["type"] != expected) {
    throw std::invalid_argument(std::string(message));
  }
}

} // namespace detail

inline void from_json(const json &j, Constant &ic) {
  detail::throw_unless_json_modifier_type(
      j, "constant", "Invalid JSON input: missing or incorrect 'type' field.");
  // Check that the JSON input has the required 'n0' field
  if (!j.contains("n0") || !j["n0"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'n0' field.");
  }
  ic.set_density(j["n0"]);
}

inline void from_json(const json &j, SingleSeed &seed) {
  detail::throw_unless_json_modifier_type(
      j, "single_seed", "JSON object does not contain a 'single_seed' type.");

  if (!j.contains("amp_eq")) {
    throw std::invalid_argument("JSON object does not contain an 'amp_eq' key.");
  }

  if (!j.contains("rho_seed")) {
    throw std::invalid_argument("JSON object does not contain an 'rho_seed' key.");
  }

  seed.set_amplitude(j["amp_eq"]);
  seed.set_density(j["rho_seed"]);
}

inline void from_json(const json &j, RandomSeeds &ic) {
  detail::throw_unless_json_modifier_type(
      j, "random_seeds", "Invalid JSON input: missing or incorrect 'type' field.");

  // Check that the JSON input has the required 'amplitude' field
  if (!j.contains("amplitude") || !j["amplitude"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'amplitude' field.");
  }

  // Check that the JSON input has the required 'rho' field
  if (!j.contains("rho") || !j["rho"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'rho' field.");
  }

  ic.set_amplitude(j["amplitude"]);
  ic.set_density(j["rho"]);
}

inline void from_json(const json &j, SeedGrid &ic) {
  detail::throw_unless_json_modifier_type(
      j, "seed_grid", "Invalid JSON input: missing or incorrect 'type' field.");

  if (!j.contains("Ny") || !j["Ny"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'Ny' field.");
  }

  if (!j.contains("Nz") || !j["Nz"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'Nz' field.");
  }

  if (!j.contains("X0") || !j["X0"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'X0' field.");
  }

  if (!j.contains("radius") || !j["radius"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'radius' field.");
  }

  if (!j.contains("amplitude") || !j["amplitude"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'amplitude' field.");
  }

  if (!j.contains("rho") || !j["rho"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'rho' field.");
  }

  ic.set_Ny(j["Ny"]);
  ic.set_Nz(j["Nz"]);
  ic.set_X0(j["X0"]);
  ic.set_radius(j["radius"]);
  ic.set_amplitude(j["amplitude"]);
  ic.set_density(j["rho"]);
}

inline void from_json(const json &j, FileReader &ic) {
  detail::throw_unless_json_modifier_type(
      j, "from_file", "Invalid JSON input: missing or incorrect 'type' field.");

  if (!j.contains("filename") || !j["filename"].is_string()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'filename' field.");
  }

  ic.set_filename(j["filename"]);
}

} // namespace pfc::ui

#endif // PFC_UI_FROM_JSON_FIELD_MODIFIERS_HPP
