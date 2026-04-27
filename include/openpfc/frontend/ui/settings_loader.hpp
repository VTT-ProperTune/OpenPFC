// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file ui/settings_loader.hpp
 * @brief Testable JSON/TOML configuration file loading helpers.
 */

#ifndef PFC_UI_SETTINGS_LOADER_HPP
#define PFC_UI_SETTINGS_LOADER_HPP

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/utils/toml_to_json.hpp>
#include <stdexcept>
#include <string>

namespace pfc::ui {

using json = nlohmann::json;

[[nodiscard]] inline json load_settings_file(const std::filesystem::path &file) {
  if (!std::filesystem::exists(file)) {
    throw std::invalid_argument("Configuration file does not exist: " +
                                file.string());
  }

  const std::string ext = file.extension().string();
  if (ext == ".toml") {
    try {
      return utils::toml_to_json(toml::parse_file(file.string()));
    } catch (const toml::parse_error &err) {
      throw std::runtime_error(
          "Error parsing TOML file: " + std::string(err.description()) +
          " at line " + std::to_string(err.source().begin.line) + ", column " +
          std::to_string(err.source().begin.column));
    }
  }

  if (ext == ".json") {
    std::ifstream input_file(file);
    if (!input_file) {
      throw std::runtime_error("Could not open configuration file: " +
                               file.string());
    }
    try {
      json settings;
      input_file >> settings;
      return settings;
    } catch (const nlohmann::json::parse_error &err) {
      throw std::runtime_error(
          "Error parsing JSON file: " + std::string(err.what()) +
          " at byte position " + std::to_string(err.byte));
    }
  }

  throw std::invalid_argument("Unsupported configuration file format: " + ext +
                              ". Supported formats: .json, .toml");
}

} // namespace pfc::ui

#endif // PFC_UI_SETTINGS_LOADER_HPP
