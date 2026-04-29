// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file ui/errors_field_modifiers.hpp
 * @brief Errors for unknown JSON field modifier / IC / BC types
 *
 * @details
 * `list_valid_field_modifiers()` is implemented in `ui_errors.cpp`. Call sites
 * that only throw on unknown modifier types can include this header without
 * pulling nlohmann JSON formatting helpers.
 *
 * @see errors_config_format.hpp
 * @see errors.hpp umbrella include
 */

#ifndef PFC_UI_ERRORS_FIELD_MODIFIERS_HPP
#define PFC_UI_ERRORS_FIELD_MODIFIERS_HPP

#include <sstream>
#include <string>
#include <vector>

namespace pfc::ui {

std::vector<std::string> list_valid_field_modifiers();

/**
 * @brief Format error for unknown field modifier type
 *
 * @see errors.hpp (historical monolithic documentation) for full contract.
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

} // namespace pfc::ui

#endif // PFC_UI_ERRORS_FIELD_MODIFIERS_HPP
