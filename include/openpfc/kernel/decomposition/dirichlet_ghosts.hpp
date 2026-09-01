// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file dirichlet_ghosts.hpp
 * @brief Node-centered Dirichlet fill of padded ghosts on one axis (M10).
 *
 * @details
 * `StagePreparationService` is the single pre-stage BC protocol. This helper
 * is the FD ghost-fill: odd reflection about the global index walls
 * `0` and `N-1` on @p axis (the non-periodic `Domain` convention:
 * nodes sit on the physical bounds). Spectral penalty BCs use the same
 * service hook with a different callable (owned-cell writes), not ghosts.
 *
 * Default order vs halo is `HaloThenBoundary` so a periodic wrap that
 * leaked onto a physical face is overwritten. `BoundaryThenHalo` is
 * appropriate when the hook mutates owned faces that neighbors must see.
 *
 * @see stage_preparation.hpp
 */

#include <stdexcept>
#include <string>

#include <openpfc/kernel/data/grid_field.hpp>

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
 * @throws std::out_of_range if the mirrored source index is not in the
 *         local padded box (halo too thin for the exterior cell).
 */
template <typename T>
inline void apply_dirichlet_ghosts(pfc::data::Field<T, pfc::HostSpace> &u, int axis,
                                   T u_low = T{}, T u_high = T{}) {
  if (axis < 0 || axis > 2) {
    throw std::invalid_argument("apply_dirichlet_ghosts: axis must be 0, 1, or 2");
  }
  const int hw = u.storage_halo();
  if (hw < 1) {
    return;
  }
  const auto lo = u.box().low;
  const auto sz = u.local_size();
  const int N = u.global_size()[axis];
  const int i0 = -hw;
  const int i1 = sz[0] + hw;
  const int j0 = -hw;
  const int j1 = sz[1] + hw;
  const int k0 = -hw;
  const int k1 = sz[2] + hw;
  for (int k = k0; k < k1; ++k) {
    for (int j = j0; j < j1; ++j) {
      for (int i = i0; i < i1; ++i) {
        const pfc::Int3 local{i, j, k};
        const int g = lo[axis] + local[axis];
        if (g >= 0 && g < N) {
          continue;
        }
        const T wall = (g < 0) ? u_low : u_high;
        const int gm = (g < 0) ? -g : (2 * (N - 1) - g);
        pfc::Int3 src = local;
        src[axis] = gm - lo[axis];
        if (src[axis] < -hw || src[axis] >= sz[axis] + hw) {
          throw std::out_of_range(
              std::string("apply_dirichlet_ghosts: mirrored index out of padded "
                          "range (axis=") +
              std::to_string(axis) + ", global=" + std::to_string(g) +
              ", mirror=" + std::to_string(gm) + ")");
        }
        u(local[0], local[1], local[2]) =
            static_cast<T>(2) * wall - u(src[0], src[1], src[2]);
      }
    }
  }
}

} // namespace pfc::communication
