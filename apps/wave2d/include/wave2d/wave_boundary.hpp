// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file wave_boundary.hpp
 * @brief Physical y-boundary ghosts (x remains periodic via MPI halo exchange).
 */

#include <openpfc/kernel/field/padded_brick.hpp>

namespace wave2d {

enum class YBoundaryKind { Dirichlet, Neumann };

/**
 * @brief Overwrite padded-brick y-halo cells that lie outside `[0, Ny)` in
 *        global y after a periodic halo exchange.
 *
 * Second-order cell-centred ghosts: partner global row `yp` in `[0, Ny-1]`.
 * **Dirichlet** (odd extension about \f$u=u_\mathrm{wall}\f$): lower
 * `yp = -1 - gj`, upper `yp = 2 * Ny - 1 - gj`,
 * `u_g = 2 u_\mathrm{wall} - u(yp)`.
 * **Neumann** (even extension): lower `yp = -gj`, upper
 * `yp = 2 * (Ny - 1) - gj`, `u_g = u(yp)`.
 */
template <class T>
inline void fill_y_physical_ghosts_padded(pfc::field::PaddedBrick<T> &u,
                                          YBoundaryKind ybc, int Ny_global,
                                          T u_wall = T{}) {
  const auto lo = u.lower_global();
  const int hw = u.halo_width();

  for (int k = -hw; k < u.nz() + hw; ++k) {
    for (int j = -hw; j < u.ny() + hw; ++j) {
      const int gj = lo[1] + j;
      if (gj >= 0 && gj < Ny_global) {
        continue;
      }
      int yp = 0;
      if (gj < 0) {
        yp = (ybc == YBoundaryKind::Dirichlet) ? (-1 - gj) : (-gj);
      } else {
        yp = (ybc == YBoundaryKind::Dirichlet) ? (2 * Ny_global - 1 - gj)
                                               : (2 * (Ny_global - 1) - gj);
      }
      const int jm = yp - lo[1];
      for (int i = -hw; i < u.nx() + hw; ++i) {
        const T um = u(i, jm, k);
        if (ybc == YBoundaryKind::Dirichlet) {
          u(i, j, k) = static_cast<T>(static_cast<T>(2) * u_wall - um);
        } else {
          u(i, j, k) = um;
        }
      }
    }
  }
}

/**
 * @brief Enforce Dirichlet \f$u = u_\mathrm{wall}\f$, \f$v = 0\f$ on owned
 *        cells at global \f$y = 0\f$ and \f$y = N_y - 1\f$.
 */
template <class T>
inline void enforce_dirichlet_y_walls_owned(pfc::field::PaddedBrick<T> &u,
                                            pfc::field::PaddedBrick<T> &v,
                                            int Ny_global, T u_wall) {
  const auto lo = u.lower_global();
  for (int k = 0; k < u.nz(); ++k) {
    for (int j = 0; j < u.ny(); ++j) {
      const int gj = lo[1] + j;
      if (gj != 0 && gj != Ny_global - 1) {
        continue;
      }
      for (int i = 0; i < u.nx(); ++i) {
        u(i, j, k) = u_wall;
        v(i, j, k) = T{0};
      }
    }
  }
}

} // namespace wave2d
