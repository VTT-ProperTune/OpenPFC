// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file dirichlet_ghosts.hpp
 * @brief Node-centered Dirichlet fill of padded ghosts on one axis (M10).
 *
 * @details
 * `StagePreparationService` is the single pre-stage BC protocol. This helper
 * is the one-axis odd mirror (`ReflectionExtent::AllOutside`, so a periodic
 * wrap on @p axis is overwritten). Node centering is the library default:
 * samples sit on the physical bounds, and this function does not pin them.
 * Cell-face centering is the `wave2d` ghost closure and also does not pin
 * owned samples. A corner that also lies outside a disabled axis is still
 * filled. Named faces, owned-node pins and stage order live in
 * `dirichlet_boundary.hpp`. Spectral penalty BCs use the same service hook
 * with a different callable (owned-cell writes).
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
#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
#include <openpfc/runtime/gpu/reflection_ghosts_gpu.hpp>
#endif

namespace pfc::communication {

/**
 * @brief Fill storage-halo cells whose global index on @p axis is outside
 *        `[0, N)`. Owned cells are not written. A corner that is also
 *        outside another axis is still filled: only @p axis is mirrored.
 *
 * Node centering mirrors layer `k` through the boundary sample:
 * `u(-k) = 2 u_low - u(k)` and `u(N-1+k) = 2 u_high - u(N-1-k)`.
 * Cell-face centering mirrors through the face outside that sample:
 * `u(-k) = 2 u_low - u(k-1)` and
 * `u(N-1+k) = 2 u_high - u(N-k)`. `wave2d` uses the cell-face closure.
 * The fill uses `ReflectionExtent::AllOutside`, so a periodic axis is
 * overwritten too. `DirichletBoundary` is the named-face operator and
 * rejects a periodic axis unless its extent says otherwise.
 *
 * Host and device fields share this function. A device field needs the
 * reflection kernel declared above; the owned wall itself is not pinned.
 *
 * @param u     Padded field (`storage_halo() >= 1`).
 * @param axis  0, 1, or 2.
 * @param u_low  Dirichlet value on the low face.
 * @param u_high Dirichlet value on the high face.
 * @param centering Node samples or cell-face samples.
 *
 * @throws std::invalid_argument if @p axis is not 0..2.
 * @throws std::out_of_range if a ghost layer does not mirror into the global
 *         index range and the local padded box.
 */
template <typename T, typename Space>
inline void
apply_dirichlet_ghosts(pfc::data::Field<T, Space> &u, int axis, T u_low = T{},
                       T u_high = T{},
                       BoundaryCentering centering = BoundaryCentering::Node) {
  if (axis < 0 || axis > 2) {
    throw std::invalid_argument("apply_dirichlet_ghosts: axis must be 0, 1, or 2");
  }
  ReflectionWalls<T> walls;
  const ReflectionSide<T> low{true, ReflectionParity::Odd, centering, u_low};
  const ReflectionSide<T> high{true, ReflectionParity::Odd, centering, u_high};
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
  apply_reflection_ghosts(u, walls, ReflectionExtent::AllOutside,
                          /*mirror_disabled_corners=*/true);
}

} // namespace pfc::communication
