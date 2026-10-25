// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <cmath>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <string>

namespace openpfc_examples {
// start snippet source
inline double coupling_source(double x, double y, double, double t) {
  return std::sin(x) * std::cos(y) + .1 * t;
}
struct HostSourceModifier {
  std::string field_name{"u"};
  void apply(pfc::SimulationState &state, double time) const {
    state.get_field<double>(field_name).apply([time](double x, double y, double z) {
      return coupling_source(x, y, z, time);
    });
  }
};
// end snippet source
} // namespace openpfc_examples
