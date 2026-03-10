// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file ui/errors.hpp
 * @brief Helpful error message formatting for JSON configuration validation
 *
 * @details
 * This module provides utility functions for generating user-friendly error
 * messages when JSON configuration validation fails. The functions format
 * error messages with:
 * - Field name and human-readable description
 * - Expected type and constraints
 * - Actual value provided by user (with type information)
 * - Valid options (when applicable)
 * - Working example syntax
 *
 * These helpers improve the user experience by making configuration errors
 * immediately actionable, reducing debugging time and support burden.
 *
 * @example
 * @code
 * // Example usage in validation code:
 * if (!j.contains("Lx") || !j["Lx"].is_number_integer()) {
 *     throw std::invalid_argument(format_config_error(
 *         "Lx",
 *         "number of grid points in X direction",
 *         "positive integer",
 *         get_json_value_string(j, "Lx"),
 *         {},
 *         "\"Lx\": 256"
 *     ));
 * }
 *
 * // Produces error message:
 * // Invalid configuration: Field 'Lx' is missing.
 * //   Description: number of grid points in X direction
 * //   Expected: positive integer
 * //   Got: missing
 * //   Example: "Lx": 256
 * @endcode
 *
 * @see ui/ui.hpp for JSON configuration interface
 *
 * @author OpenPFC Development Team
 * @date 2025-11-25
 */

#ifndef PFC_UI_ERRORS_HPP
#define PFC_UI_ERRORS_HPP

#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace pfc {
namespace ui {

/**
 * @brief Format a helpful configuration error message
 *
 * Creates a multi-line error message with complete context about a
 * configuration validation failure. The message includes:
 * - Field name and what it represents
 * - Expected type/format with constraints
 * - Actual value provided (or "missing")
 * - List of valid options (if applicable)
 * - Working example syntax (if provided)
 *
 * Error messages are designed to be:
 * - **Specific**: Explain exactly what's wrong
 * - **Contextual**: Show what user provided vs. what's expected
 * - **Actionable**: List valid options and provide examples
 * - **Concise**: Typically < 10 lines
 *
 * @param field_name Name of problematic field (e.g., "Lx", "origo")
 * @param description Human-readable description of field purpose
 * @param expected_type Expected type with constraints (e.g., "positive integer")
 * @param actual_value What user provided (use get_json_value_string())
 * @param valid_options List of valid values (empty if not applicable)
 * @param example Working example syntax (empty if not needed)
 * @return Formatted multi-line error message ready for exception
 *
 * @note This is a pure function with no side effects
 * @note Thread-safe
 *
 * Time complexity: O(n) where n = number of valid_options
 *
 * @code
 * // Example: Missing field
 * auto msg = format_config_error(
 *     "Lx", "grid points in X", "positive integer",
 *     "missing", {}, "\"Lx\": 256"
 * );
 * // Output:
 * // Invalid configuration: Field 'Lx' is missing.
 * //   Description: grid points in X
 * //   Expected: positive integer
 * //   Got: missing
 * //   Example: "Lx": 256
 *
 * // Example: Invalid enum value
 * auto msg = format_config_error(
 *     "origin", "coordinate system origin", "string",
 *     "\"centre\"", {"center", "corner"}, "\"origin\": \"center\""
 * );
 * // Output:
 * // Invalid configuration: Field 'origin' has invalid value.
 * //   Description: coordinate system origin
 * //   Expected: string
 * //   Got: "centre"
 * //   Valid options: 'center', 'corner'
 * //   Example: "origo": "center"
 * @endcode
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
      if (i < valid_options.size() - 1) oss << ", ";
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
 *
 * Extracts the actual value from a JSON object and formats it for display
 * in error messages. Returns "missing" if field doesn't exist, otherwise
 * returns the JSON representation with type information.
 *
 * Type information helps users understand type mismatches:
 * - integer: `42 (type: integer)`
 * - float: `256.5 (type: float)`
 * - string: `"hello" (type: string)`
 * - boolean: `true (type: boolean)`
 * - null: `null (type: null)`
 * - array: `[1,2,3] (type: array)`
 * - object: `{"key":"value"} (type: object)`
 *
 * @param j JSON object to extract value from
 * @param field_name Field name to look up
 * @return Formatted string describing the value with type
 *
 * @note This is a pure function with no side effects
 * @note Thread-safe
 *
 * Time complexity: O(n) where n = size of JSON value when dumping
 *
 * @code
 * json j = {{"Lx", 256.5}, {"origin", "center"}};
 *
 * auto lx = get_json_value_string(j, "Lx");
 * // Returns: "256.5 (type: float)"
 *
 * auto origin = get_json_value_string(j, "origin");
 * // Returns: "\"center\" (type: string)"
 *
 * auto missing = get_json_value_string(j, "nonexistent");
 * // Returns: "missing"
 * @endcode
 */
inline std::string get_json_value_string(const nlohmann::json &j,
                                         const std::string &field_name) {
  if (!j.contains(field_name)) {
    return "missing";
  }

  const auto &value = j[field_name];
  std::ostringstream oss;
  oss << value.dump(); // JSON representation

  // Add type information
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

/**
 * @brief List all registered field modifiers
 *
 * Returns a vector of all field modifier type names that can be used in
 * JSON configuration. Includes both initial conditions and boundary conditions.
 *
 * Initial conditions:
 * - "constant": Uniform field value
 * - "single_seed": Single crystalline seed
 * - "random_seeds": Random seed pattern
 * - "seed_grid": Regular grid of seeds
 * - "from_file": Load from binary file
 *
 * Boundary conditions:
 * - "fixed": Fixed boundary values
 * - "moving": Moving boundary
 *
 * @return Vector of registered type names
 *
 * @note This function is defined in ui_errors.cpp
 * @note Must be kept in sync with FieldModifierInitializer in ui.hpp
 *
 * @warning This function is not yet extensible. Custom field modifiers
 *          registered by users are not automatically included in this list.
 *          Future enhancement: FieldModifierRegistry::get_registered_types()
 *
 * Time complexity: O(1) - returns fixed list
 */
std::vector<std::string> list_valid_field_modifiers();

/**
 * @brief Format error for unknown field modifier type
 *
 * Creates a helpful error message when user specifies an unknown field
 * modifier type in JSON configuration. Lists all valid types to help
 * user correct the spelling or choose the right modifier.
 *
 * @param invalid_type The type user tried to use
 * @param context Context string (e.g., "field modifier", "initial condition")
 * @return Formatted error message with valid types listed
 *
 * @note Calls list_valid_field_modifiers() to get current list
 * @note Thread-safe
 *
 * Time complexity: O(n) where n = number of registered field modifiers
 *
 * @code
 * // Example: Unknown modifier type
 * auto msg = format_unknown_modifier_error("random_seed");
 * // Output:
 * // Unknown field modifier type: 'random_seed'
 * //   Valid types:
 * //     - constant
 * //     - single_seed
 * //     - random_seeds
 * //     - seed_grid
 * //     - from_file
 * //     - fixed
 * //     - moving
 * //   Check spelling and see documentation for details.
 *
 * // Example: With custom context
 * auto msg = format_unknown_modifier_error("foo", "initial condition");
 * // Output:
 * // Unknown initial condition type: 'foo'
 * //   Valid types: ...
 * @endcode
 */
inline std::string
format_unknown_modifier_error(const std::string &invalid_type,
                              const std::string &context = "field modifier") {
  auto valid_types = list_valid_field_modifiers();

  std::ostringstream oss;
  oss << "Unknown " << context << " type: '" << invalid_type << "'\n";
  oss << "  Valid types:\n";
  for (const auto &type : valid_types) {
    oss << "    - " << type << "\n";
  }
  oss << "  Check spelling and see documentation for details.";

  return oss.str();
}

} // namespace ui
} // namespace pfc

#endif // PFC_UI_ERRORS_HPP

