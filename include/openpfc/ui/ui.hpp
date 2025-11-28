// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file ui/ui.hpp
 * @brief Main UI header - includes all UI components
 *
 * @details
 * This header provides the main entry point for the UI system. It includes
 * all the refactored UI components:
 * - JSON helpers for configuration parsing
 * - JSON deserialization functions
 * - Field modifier registry
 * - Application class
 * - Error handling utilities
 *
 * This header maintains backward compatibility with the original ui.hpp
 * by including all the split components.
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef PFC_UI_UI_HPP
#define PFC_UI_UI_HPP

#include "app.hpp"
#include "errors.hpp"
#include "field_modifier_registry.hpp"
#include "from_json.hpp"
#include "json_helpers.hpp"

#endif // PFC_UI_UI_HPP
