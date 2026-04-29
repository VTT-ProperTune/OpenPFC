// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file ui/errors_config_format.hpp
 * @brief JSON-oriented configuration error formatting (`format_config_error`, value
 *        introspection)
 *
 * @details
 * Parsers that only need multi-line field errors and `get_json_value_string` can
 * include this header without pulling field-modifier catalog symbols.
 *
 * @see errors_field_modifiers.hpp for unknown modifier type messages
 * @see errors.hpp umbrella include
 */

#ifndef PFC_UI_ERRORS_CONFIG_FORMAT_HPP
#define PFC_UI_ERRORS_CONFIG_FORMAT_HPP

#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace pfc::ui {

/**
 * @brief Format a helpful configuration error message
 *
 * @see errors.hpp (historical monolithic documentation) for full contract and
 *      examples.
 */
inline std::string
format_config_error(const std::string &field_name, const std::string &description,
                    const std::string &expected_type,
                    const std::string &actual_value,
                    const std::vector<std::string> &valid_options = {},
                    const std::string &example = "") {
  std::ostringstream oss;
  oss << "Invalid configuration: Field '" << field_name << "' ";

  if (actual_value == "missing") {
    oss << "is missing.\n";
  } else {
    oss << "has invalid value.\n";
  }

  oss << "  Description: " << description << "\n";
  oss << "  Expected: " << expected_type << "\n";
  oss << "  Got: " << actual_value << "\n";

  if (!valid_options.empty()) {
    oss << "  Valid options: ";
    for (size_t i = 0; i < valid_options.size(); ++i) {
      oss << "'" << valid_options[i] << "'";
      if (i < valid_options.size() - 1) {
        oss << ", ";
      }
    }
    oss << "\n";
  }

  if (!example.empty()) {
    oss << "  Example: " << example;
  }

  return oss.str();
}

/**
 * @brief Get JSON value as string for error messages
 */
inline std::string get_json_value_string(const nlohmann::json &j,
                                         const std::string &field_name) {
  if (!j.contains(field_name)) {
    return "missing";
  }

  const auto &value = j[field_name];
  std::ostringstream oss;
  oss << value.dump(); // JSON representation

  if (value.is_null()) {
    oss << " (type: null)";
  } else if (value.is_boolean()) {
    oss << " (type: boolean)";
  } else if (value.is_number_integer()) {
    oss << " (type: integer)";
  } else if (value.is_number_float()) {
    oss << " (type: float)";
  } else if (value.is_string()) {
    oss << " (type: string)";
  } else if (value.is_array()) {
    oss << " (type: array)";
  } else if (value.is_object()) {
    oss << " (type: object)";
  }

  return oss.str();
}

} // namespace pfc::ui

#endif // PFC_UI_ERRORS_CONFIG_FORMAT_HPP
