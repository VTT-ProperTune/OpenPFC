// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file ui/json_helpers.hpp
 * @brief JSON utility functions for configuration parsing
 *
 * @details
 * This header provides helper functions for working with JSON configuration
 * objects, particularly for handling both flat and nested JSON structures.
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef PFC_UI_JSON_HELPERS_HPP
#define PFC_UI_JSON_HELPERS_HPP

#include <nlohmann/json.hpp>
#include <string>

namespace pfc {
namespace ui {

using json = nlohmann::json;

/**
 * @brief Helper function to get a JSON value from either flat or nested structure
 *
 * Supports both flat access (j["Lx"]) and nested access (j["domain"]["Lx"]).
 * Checks nested location first if the key doesn't exist at top level.
 *
 * @param j JSON object to search
 * @param key Key to look for
 * @param section Optional section name (e.g., "domain", "timestepping")
 * @return JSON value if found, otherwise json(nullptr)
 */
inline json get_json_value(const json &j, const std::string &key,
                           const std::string &section = "") {
  // First try direct access
  if (j.contains(key)) {
    return j[key];
  }
  // If section is specified, try nested access
  if (!section.empty() && j.contains(section) && j[section].is_object() &&
      j[section].contains(key)) {
    return j[section][key];
  }
  // Try common sections if not specified
  if (section.empty()) {
    // Try domain section for domain-related keys
    if (j.contains("domain") && j["domain"].is_object() &&
        j["domain"].contains(key)) {
      return j["domain"][key];
    }
    // Try timestepping section for time-related keys
    if (j.contains("timestepping") && j["timestepping"].is_object() &&
        j["timestepping"].contains(key)) {
      return j["timestepping"][key];
    }
  }
  return json(nullptr);
}

} // namespace ui
} // namespace pfc

#endif // PFC_UI_JSON_HELPERS_HPP
