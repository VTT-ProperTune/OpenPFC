// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/frontend/ui/field_modifier_registry.hpp>

namespace pfc::ui {

std::vector<std::string> list_valid_field_modifiers() {
  return default_field_modifier_catalog().registered_modifier_types();
}

} // namespace pfc::ui
