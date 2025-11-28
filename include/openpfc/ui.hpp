// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file ui.hpp
 * @brief JSON-based configuration interface (backward compatibility header)
 *
 * @details
 * This header provides backward compatibility for code that includes
 * <openpfc/ui.hpp>. The UI code has been refactored into smaller modules
 * in the ui/ subdirectory. This header simply includes the new structure.
 *
 * @note For new code, consider including specific headers from ui/ directly:
 * - ui/json_helpers.hpp - JSON utility functions
 * - ui/from_json.hpp - JSON deserialization
 * - ui/field_modifier_registry.hpp - Field modifier registry
 * - ui/app.hpp - Application class
 * - ui/errors.hpp - Error handling utilities
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef PFC_UI_HPP
#define PFC_UI_HPP

// Include the refactored UI components
#include "ui/ui.hpp"

#endif // PFC_UI_HPP
