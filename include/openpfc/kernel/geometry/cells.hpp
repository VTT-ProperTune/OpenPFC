// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file cells.hpp
 * @brief Cartesian cells on an existing `Domain`.
 *
 * Nodes stay `origin + index * spacing`. A periodic axis has one cell per
 * node; the last cell ends on the periodic image of the origin. A
 * non-periodic axis has one fewer cell than nodes. An axis with a single
 * node is inactive: the measure does not multiply by that spacing, which
 * is how a 2D domain (`nz == 1`) reports area.
 */

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/geometry/vec3.hpp>

#include <stdexcept>

namespace pfc::geometry {

struct Box {
  Vec3 lo{};
  Vec3 hi{};
};

struct CellLayout {
  int count[3] = {1, 1, 1};
  int active[3] = {1, 1, 1};
  int periodic[3] = {0, 0, 0};
  Vec3 origin{};
  Vec3 spacing{1, 1, 1};

  OPENPFC_HD int size() const { return count[0] * count[1] * count[2]; }

  OPENPFC_HD int linear(int i, int j, int k) const {
    return i + count[0] * (j + count[1] * k);
  }

  OPENPFC_HD int dimension() const { return active[0] + active[1] + active[2]; }

  OPENPFC_HD Box box(int i, int j, int k) const {
    const int idx[3] = {i, j, k};
    Box out;
    double *lo = &out.lo.x;
    double *hi = &out.hi.x;
    const double *o = &origin.x;
    const double *h = &spacing.x;
    for (int a = 0; a < 3; ++a) {
      lo[a] = o[a] + idx[a] * h[a];
      hi[a] = active[a] ? lo[a] + h[a] : lo[a];
    }
    return out;
  }

  OPENPFC_HD double measure(Box cell) const {
    double m = 1;
    const double *lo = &cell.lo.x;
    const double *hi = &cell.hi.x;
    for (int a = 0; a < 3; ++a)
      if (active[a]) m *= hi[a] - lo[a];
    return m;
  }

  OPENPFC_HD Vec3 period() const {
    return {count[0] * spacing.x, count[1] * spacing.y, count[2] * spacing.z};
  }
};

inline CellLayout layout_from_domain(const pfc::Domain &domain) {
  CellLayout layout;
  layout.origin = {domain.origin[0], domain.origin[1], domain.origin[2]};
  layout.spacing = {domain.spacing[0], domain.spacing[1], domain.spacing[2]};
  for (int a = 0; a < 3; ++a) {
    if (domain.size[a] <= 1) {
      layout.count[a] = 1;
      layout.active[a] = 0;
    } else if (domain.periodic[a]) {
      layout.count[a] = domain.size[a];
      layout.active[a] = 1;
    } else {
      layout.count[a] = domain.size[a] - 1;
      layout.active[a] = 1;
    }
    layout.periodic[a] = domain.periodic[a] ? 1 : 0;
    if (layout.count[a] < 1)
      throw std::invalid_argument("cell layout needs a positive count");
  }
  if (layout.dimension() < 1)
    throw std::invalid_argument("cell layout needs an active axis");
  return layout;
}

} // namespace pfc::geometry
