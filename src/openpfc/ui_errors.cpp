// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/ui_errors.hpp"

namespace pfc {
namespace ui {

std::vector<std::string> list_valid_field_modifiers() {
  // This needs to be kept in sync with FieldModifierInitializer in ui.hpp
  // Future enhancement: Make FieldModifierRegistry provide this method
  // to avoid manual synchronization
  return {// Initial conditions
          "constant", "single_seed", "random_seeds", "seed_grid", "from_file",
          // Boundary conditions
          "fixed", "moving"};
}

} // namespace ui
} // namespace pfc
