// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once
// Deliberate app-owned copy: these are solidification-model rules, not a
// physics-agnostic OpenPFC API. Independently buildable apps cannot include a
// sibling app. Share them only after extraction to their model's software owner;
// until then retain parity with the other solidification app's boundary rules.

/**
 * @file aluminum/solidification_bc_json.hpp
 * @brief JSON `from_json` and catalog registration for app-local FixedBC/MovingBC.
 */

#include <stdexcept>

#include <aluminum/fixed_bc.hpp>
#include <aluminum/moving_bc.hpp>
#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/ui/from_json_field_modifiers.hpp>

namespace aluminum {

// `from_json` lives next to the type so ADL finds it from
// `pfc::ui::register_field_modifier`.
inline void from_json(const pfc::ui::json &j, FixedBC &bc) {
  pfc::ui::detail::throw_unless_json_modifier_type(
      j, "fixed", "Invalid JSON input: missing or incorrect 'type' field.");

  if (!j.contains("rho_low") || !j["rho_low"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'rho_low' field.");
  }
  if (!j.contains("rho_high") || !j["rho_high"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'rho_high' field.");
  }
  bc.set_rho_low(j["rho_low"]);
  bc.set_rho_high(j["rho_high"]);
}

inline void from_json(const pfc::ui::json &j, MovingBC &bc) {
  pfc::ui::detail::throw_unless_json_modifier_type(
      j, "moving", "Invalid JSON input: missing or incorrect 'type' field.");

  if (!j.contains("rho_low") || !j["rho_low"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'rho_low' field.");
  }
  if (!j.contains("rho_high") || !j["rho_high"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'rho_high' field.");
  }
  if (!j.contains("width") || !j["width"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'width' field.");
  }
  if (!j.contains("alpha") || !j["alpha"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'alpha' field.");
  }
  if (!j.contains("disp") || !j["disp"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'disp' field.");
  }
  if (!j.contains("xpos") || !j["xpos"].is_number()) {
    throw std::invalid_argument(
        "Invalid JSON input: missing or invalid 'xpos' field.");
  }
  bc.set_rho_low(j["rho_low"]);
  bc.set_rho_high(j["rho_high"]);
  bc.set_xwidth(j["width"]);
  bc.set_alpha(j["alpha"]);
  bc.set_disp(j["disp"]);
  bc.set_xpos(j["xpos"]);
}

/** Register JSON `"fixed"` / `"moving"` on the process-wide catalog. */
inline void register_solidification_bcs() {
  pfc::ui::register_field_modifier<FixedBC>("fixed");
  pfc::ui::register_field_modifier<MovingBC>("moving");
}

} // namespace aluminum
