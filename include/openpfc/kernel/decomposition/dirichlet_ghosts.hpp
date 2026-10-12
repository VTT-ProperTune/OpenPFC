// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file dirichlet_ghosts.hpp
 * @brief Node-centered Dirichlet fill of padded ghosts on one axis (M10).
 *
 * @details
 * `StagePreparationService` is the single pre-stage BC protocol. This helper
 * is the one-axis node-centered odd case of `apply_reflection_ghosts`
 * (`ReflectionExtent::AllOutside`, so a periodic wrap on @p axis is
 * overwritten). Nodes sit on the physical bounds. Spectral penalty BCs use
 * the same service hook with a different callable (owned-cell writes).
 *
 * Default order vs halo is `HaloThenBoundary` so a periodic wrap that
 * leaked onto a physical face is overwritten. `BoundaryThenHalo` is
 * appropriate when the hook mutates owned faces that neighbors must see.
 *
 * @see stage_preparation.hpp
 * @see reflection_ghosts.hpp
 */

#include <stdexcept>

#include <openpfc/kernel/decomposition/reflection_ghosts.hpp>

namespace pfc::communication {

/**
 * @brief Fill storage-halo cells whose global index on @p axis is outside
 *        `[0, N)`. Owned cells are not written.
 *
 * @param u     Padded host field (`storage_halo() >= 1`).
 * @param axis  0, 1, or 2.
 * @param u_low  Dirichlet value at global index 0.
 * @param u_high Dirichlet value at global index `N-1`.
 *
 * @throws std::invalid_argument if @p axis is not 0..2.
 * @throws std::out_of_range if a ghost layer does not mirror into the global
 *         index range and the local padded box.
 */
template <typename T>
inline void apply_dirichlet_ghosts(pfc::data::Field<T, pfc::HostSpace> &u, int axis,
                                   T u_low = T{}, T u_high = T{}) {
  if (axis < 0 || axis > 2) {
    throw std::invalid_argument("apply_dirichlet_ghosts: axis must be 0, 1, or 2");
  }
  ReflectionWalls<T> walls;
  const ReflectionSide<T> low{true, ReflectionParity::Odd, BoundaryCentering::Node,
                              u_low};
  const ReflectionSide<T> high{true, ReflectionParity::Odd, BoundaryCentering::Node,
                               u_high};
  if (axis == 0) {
    walls.xm = low;
    walls.xp = high;
  } else if (axis == 1) {
    walls.ym = low;
    walls.yp = high;
  } else {
    walls.zm = low;
    walls.zp = high;
  }
  apply_reflection_ghosts(u, walls, ReflectionExtent::AllOutside);
}

} // namespace pfc::communication
