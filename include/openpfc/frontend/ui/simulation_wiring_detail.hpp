// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulation_wiring_detail.hpp
 * @brief Shared JSON `target` parsing for IC/BC field modifiers
 */

#ifndef PFC_UI_SIMULATION_WIRING_DETAIL_HPP
#define PFC_UI_SIMULATION_WIRING_DETAIL_HPP

#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>
#include <openpfc/kernel/utils/logging.hpp>

namespace pfc::ui {
namespace detail {

/**
 * @brief Parse JSON `target` (string or array) and apply to a field modifier
 *
 * Single place for initial-condition and boundary-condition wiring (DRY).
 * @param modifier_kind Log label, e.g. `"initial condition"` or `"boundary
 * condition"`.
 */
inline void configure_field_modifier_targets_from_json(
    pfc::FieldModifier &modifier, const nlohmann::json &params,
    const pfc::Logger &lg, bool rank0, std::string_view modifier_kind) {
  if (!params.contains("target")) {
    if (rank0) {
      pfc::log_warning(lg, std::string("no target is set for ") +
                               std::string(modifier_kind) +
                               "! Using target 'default'");
    }
    return;
  }
  const auto &target = params["target"];
  if (target.is_array()) {
    std::vector<std::string> names;
    names.reserve(target.size());
    for (const auto &el : target) {
      names.push_back(el.get<std::string>());
    }
    modifier.set_field_names(std::move(names));
    if (rank0) {
      pfc::log_info(lg, std::string("Setting ") + std::string(modifier_kind) +
                            " targets (multi-field)");
    }
    return;
  }
  const auto t = target.get<std::string>();
  if (rank0) {
    pfc::log_info(lg, std::string("Setting ") + std::string(modifier_kind) +
                          " target to " + t);
  }
  modifier.set_field_name(t);
}

} // namespace detail
} // namespace pfc::ui

#endif // PFC_UI_SIMULATION_WIRING_DETAIL_HPP
