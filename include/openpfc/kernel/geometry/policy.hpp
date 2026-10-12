// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file policy.hpp
 * @brief Tolerances for implicit interfaces on a Cartesian cell.
 *
 * `length_*` is a physical length. It is converted to a band in the
 * implicit function by multiplying by `||∇φ||`, so a signed-distance
 * field (where that factor is 1) and an ellipsoid polynomial (where it
 * is not) share one geometric meaning.
 *
 * A positive volume below `sliver_fraction` of the cell is kept and
 * flagged. It is not discarded.
 */

#include <openpfc/kernel/data/host_device.hpp>

namespace pfc::geometry {

struct Policy {
  double length_absolute = 1e-12;
  double length_relative = 1e-12;
  double gradient_absolute = 1e-14;
  double sliver_fraction = 1e-10;
  int max_depth = 6;
  int moment_degree = 2;
};

OPENPFC_HD inline double length_tolerance(const Policy &p, double length_scale) {
  return p.length_absolute + p.length_relative * length_scale;
}

/// Band in φ corresponding to `length` at a point whose gradient has `grad`.
OPENPFC_HD inline double phi_band(const Policy &p, double grad,
                                  double length_scale) {
  const double g = grad > p.gradient_absolute ? grad : p.gradient_absolute;
  return length_tolerance(p, length_scale) * g;
}

enum class Side { Inside, Outside, OnInterface };

enum class CellKind { Interior, Exterior, Cut };

} // namespace pfc::geometry
