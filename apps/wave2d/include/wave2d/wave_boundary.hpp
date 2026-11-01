// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file wave_boundary.hpp
 * @brief Physical y-boundary ghosts (x remains periodic via MPI halo exchange).
 */

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/dirichlet_ghosts.hpp>
#include <stdexcept>

namespace wave2d {

enum class YBoundaryKind { Dirichlet, Neumann };

/**
 * @brief Overwrite y-halo cells that lie outside `[0, Ny)` in
 *        global y after a periodic halo exchange.
 *
 * Dirichlet ghosts are the kernel cell-face odd mirror
 * (`apply_dirichlet_ghosts` on axis y). That closure does not pin owned
 * samples. `enforce_dirichlet_y_walls_owned` is a separate wave update:
 * it sets the endpoint rows of `u` and `v`, which this grid stores as
 * owned samples even though the ghost formula is cell-face centred.
 * Neumann keeps the even node extension below.
 */
template <class T>
inline void fill_y_physical_ghosts_padded(pfc::data::Field<T, pfc::HostSpace> &u,
                                          YBoundaryKind ybc, int Ny_global,
                                          T u_wall = T{}) {
  if (ybc == YBoundaryKind::Dirichlet) {
    pfc::communication::apply_dirichlet_ghosts(
        u, /*axis=*/1, u_wall, u_wall,
        pfc::communication::BoundaryCentering::CellFace);
    return;
  }
  const auto lo = u.box().low;
  const int hw = u.halo_width();
  const auto sz = u.local_size();

  for (int k = -hw; k < sz[2] + hw; ++k) {
    for (int j = -hw; j < sz[1] + hw; ++j) {
      const int gj = lo[1] + j;
      if (gj >= 0 && gj < Ny_global) {
        continue;
      }
      const int yp = (gj < 0) ? (-gj) : (2 * (Ny_global - 1) - gj);
      const int jm = yp - lo[1];

      // Bounds check: mirrored local index must be within valid padded range
      if (jm < -hw || jm >= sz[1] + hw) {
        throw std::out_of_range(
            "fill_y_physical_ghosts_padded: mirrored ghost index out of bounds. "
            "halo_width=" +
            std::to_string(hw) + ", local_ny=" + std::to_string(sz[1]) +
            ", valid_range=[" + std::to_string(-hw) + "," +
            std::to_string(sz[1] + hw) +
            "), mirrored_global_yp=" + std::to_string(yp) +
            ", computed_local_jm=" + std::to_string(jm) + ", boundary_kind=Neumann");
      }

      for (int i = -hw; i < sz[0] + hw; ++i) {
        u(i, j, k) = u(i, jm, k);
      }
    }
  }
}

/**
 * @brief Enforce Dirichlet \f$u = u_\mathrm{wall}\f$, \f$v = 0\f$ on owned
 *        cells at global \f$y = 0\f$ and \f$y = N_y - 1\f$.
 */
template <class T>
inline void enforce_dirichlet_y_walls_owned(pfc::data::Field<T, pfc::HostSpace> &u,
                                            pfc::data::Field<T, pfc::HostSpace> &v,
                                            int Ny_global, T u_wall) {
  const auto lo = u.box().low;
  const auto sz = u.local_size();
  for (int k = 0; k < sz[2]; ++k) {
    for (int j = 0; j < sz[1]; ++j) {
      const int gj = lo[1] + j;
      if (gj != 0 && gj != Ny_global - 1) {
        continue;
      }
      for (int i = 0; i < sz[0]; ++i) {
        u(i, j, k) = u_wall;
        v(i, j, k) = T{0};
      }
    }
  }
}

} // namespace wave2d
