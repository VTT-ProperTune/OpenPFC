// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file ui/errors.hpp
 * @brief Umbrella include for UI configuration error helpers
 *
 * @details
 * Split by concern so lightweight parsers avoid unrelated dependencies:
 * - @ref errors_config_format.hpp — `format_config_error`, `get_json_value_string`
 * - @ref errors_field_modifiers.hpp — `list_valid_field_modifiers`,
 *   `format_unknown_modifier_error`
 *
 * @see ui/ui.hpp for JSON configuration interface
 */

#ifndef PFC_UI_ERRORS_HPP
#define PFC_UI_ERRORS_HPP

#include <openpfc/frontend/ui/errors_config_format.hpp>
#include <openpfc/frontend/ui/errors_field_modifiers.hpp>

#endif // PFC_UI_ERRORS_HPP
