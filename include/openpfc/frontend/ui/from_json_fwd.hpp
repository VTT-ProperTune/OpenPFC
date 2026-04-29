// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file from_json_fwd.hpp
 * @brief Primary `from_json<T>` declaration for UI JSON parsing
 */

#ifndef PFC_UI_FROM_JSON_FWD_HPP
#define PFC_UI_FROM_JSON_FWD_HPP

#include <openpfc/frontend/ui/json_helpers.hpp>

namespace pfc::ui {

template <class T> [[nodiscard]] T from_json(const json &j);

} // namespace pfc::ui

#endif // PFC_UI_FROM_JSON_FWD_HPP
