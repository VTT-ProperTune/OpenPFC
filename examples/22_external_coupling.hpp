// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file 22_external_coupling.hpp
 * @brief Shared source formula for the external-coupling example and tests.
 *
 * FieldModifier-shaped adapter: `apply(SimulationState&, double)` writes a
 * time-varying source into a named host field. The FEM side reads geometry
 * through `pfc::coupling::FieldHandle` and writes back via this adapter.
 */

#include <cmath>
#include <string>

#include <openpfc/kernel/simulation/simulation_state.hpp>

namespace openpfc_examples {

[[nodiscard]] inline double coupling_source(double x, double y, double /*z*/,
                                            double t) {
  return std::sin(x) * std::cos(y) + 0.1 * t;
}

/**
 * @brief FieldModifier-shaped host source (binds `SimulationState`, not Gen-1
 *        `Model`).
 */
struct HostSourceModifier {
  std::string field_name{"u"};

  void apply(pfc::SimulationState &state, double time) const {
    auto &f = state.get_field<double>(field_name);
    f.apply([time](double x, double y, double z) {
      return coupling_source(x, y, z, time);
    });
  }
};

} // namespace openpfc_examples
