// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulation_wiring_conditions.hpp
 * @brief JSON-driven initial and boundary conditions
 */

#ifndef PFC_UI_SIMULATION_WIRING_CONDITIONS_HPP
#define PFC_UI_SIMULATION_WIRING_CONDITIONS_HPP

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/ui/errors_config_format.hpp>
#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/ui/simulation_wiring_context.hpp>
#include <openpfc/frontend/ui/simulation_wiring_detail.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>
#include <openpfc/kernel/utils/logging.hpp>

namespace pfc::ui {
namespace detail {

/**
 * @brief Shared loop for JSON `initial_conditions` / `boundary_conditions` arrays
 *
 * Keeps logging strings, type checks, factory call, and target wiring in one
 * place (DRY); `RegisterFn` receives each constructed `FieldModifier`.
 */
template <typename RegisterFn>
void wire_field_modifiers_from_json_array(
    const nlohmann::json &settings, const JsonWiringContext &ctx,
    const FieldModifierCatalog &modifier_catalog, const char *json_array_key,
    std::string_view empty_section_warning, std::string_view section_header_log,
    std::string_view creating_log_prefix, std::string_view modifier_kind_label,
    RegisterFn &&register_modifier) {
  const pfc::Logger lg{pfc::LogLevel::Info, ctx.mpi_rank};
  if (!settings.contains(json_array_key)) {
    if (ctx.rank0) {
      pfc::log_warning(lg, std::string(empty_section_warning));
    }
    return;
  }
  if (ctx.rank0) {
    pfc::log_info(lg, std::string(section_header_log));
  }
  for (const nlohmann::json &params : settings[json_array_key]) {
    if (ctx.rank0) {
      std::ostringstream ps;
      ps << params;
      pfc::log_info(lg, std::string(creating_log_prefix) + ps.str());
    }
    if (!params.contains("type") || !params["type"].is_string()) {
      throw std::invalid_argument(format_config_error(
          "type", std::string(modifier_kind_label) + " type",
          "string (a registered catalog key)", get_json_value_string(params, "type"),
          modifier_catalog.registered_modifier_types(), "\"type\": \"constant\""));
    }
    std::string type = params["type"].get<std::string>();
    auto field_modifier = create_field_modifier(type, params, modifier_catalog);
    configure_field_modifier_targets_from_json(*field_modifier, params, lg,
                                               ctx.rank0, modifier_kind_label);
    field_modifier->set_mpi_comm(ctx.comm);
    register_modifier(std::move(field_modifier));
  }
}

} // namespace detail

[[nodiscard]] inline std::vector<std::unique_ptr<FieldModifier>>
parse_initial_conditions_from_json(const nlohmann::json &settings,
                                   const JsonWiringContext &ctx,
                                   const FieldModifierCatalog &modifier_catalog) {
  std::vector<std::unique_ptr<FieldModifier>> out;
  detail::wire_field_modifiers_from_json_array(
      settings, ctx, modifier_catalog, "initial_conditions",
      "no initial conditions are set!", "Adding initial conditions",
      "Creating initial condition from data ", "initial condition",
      [&](std::unique_ptr<FieldModifier> m) { out.push_back(std::move(m)); });
  return out;
}

[[nodiscard]] inline std::vector<std::unique_ptr<FieldModifier>>
parse_boundary_conditions_from_json(const nlohmann::json &settings,
                                    const JsonWiringContext &ctx,
                                    const FieldModifierCatalog &modifier_catalog) {
  std::vector<std::unique_ptr<FieldModifier>> out;
  detail::wire_field_modifiers_from_json_array(
      settings, ctx, modifier_catalog, "boundary_conditions",
      "no boundary conditions are set!", "Adding boundary conditions",
      "Creating boundary condition from data ", "boundary condition",
      [&](std::unique_ptr<FieldModifier> m) { out.push_back(std::move(m)); });
  return out;
}

} // namespace pfc::ui

#endif // PFC_UI_SIMULATION_WIRING_CONDITIONS_HPP
