// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file initial_condition.hpp
 * @brief Apply a `PointFn` initial condition to a local field.
 *
 * Two helpers — one per backend — that walk every local cell, translate the
 * local index to a physical coordinate via the world's `origin/spacing`, and
 * write `ic(x, y, z)` into the field. The model holds the lambda; these
 * helpers do the index walking the framework knows about.
 */

#include <cstddef>
#include <vector>

#include <heat3d/heat_model.hpp>

#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/box3i.hpp>

namespace heat3d {

/**
 * @brief Fill `u` (sized to the local FD subdomain) by sampling `ic` at
 *        every owned cell of `rank`.
 *
 * Resizes `u` to `nx*ny*nz` before writing, matching the existing heat3d FD
 * convention.
 */
inline void
apply_initial_condition_fd(std::vector<double> &u,
                           const pfc::decomposition::Decomposition &decomp, int rank,
                           const PointFn &ic) {
  const auto &gw = pfc::decomposition::get_world(decomp);
  const auto &local = pfc::decomposition::get_subworld(decomp, rank);
  auto lo = pfc::world::get_lower(local);
  auto sz = pfc::world::get_size(local);
  const int nx = sz[0];
  const int ny = sz[1];
  const int nz = sz[2];
  const int sxy = nx * ny;
  const auto origin = pfc::world::get_origin(gw);
  const auto spacing = pfc::world::get_spacing(gw);
  u.assign(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
               static_cast<std::size_t>(nz),
           0.0);
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const int gi = lo[0] + ix;
        const int gj = lo[1] + iy;
        const int gk = lo[2] + iz;
        const double x = origin[0] + static_cast<double>(gi) * spacing[0];
        const double y = origin[1] + static_cast<double>(gj) * spacing[1];
        const double z = origin[2] + static_cast<double>(gk) * spacing[2];
        const std::size_t idx =
            static_cast<std::size_t>(ix) +
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(iz) * static_cast<std::size_t>(sxy);
        u[idx] = ic(x, y, z);
      }
    }
  }
}

/**
 * @brief Fill `u` (sized to the FFT inbox) by sampling `ic` at every cell of
 *        `inbox` for this rank.
 *
 * `u` must already be sized to `fft.size_inbox()`; the order of cells matches
 * the inbox iteration `for k in [low.z..high.z]` etc.
 */
inline void
apply_initial_condition_spectral(std::vector<double> &u,
                                 const pfc::decomposition::Decomposition &decomp,
                                 const pfc::fft::Box3i &inbox, const PointFn &ic) {
  const auto &gw = pfc::decomposition::get_world(decomp);
  const auto origin = pfc::world::get_origin(gw);
  const auto spacing = pfc::world::get_spacing(gw);
  std::size_t idx = 0;
  for (int k = inbox.low[2]; k <= inbox.high[2]; ++k) {
    for (int j = inbox.low[1]; j <= inbox.high[1]; ++j) {
      for (int i = inbox.low[0]; i <= inbox.high[0]; ++i) {
        const double x = origin[0] + static_cast<double>(i) * spacing[0];
        const double y = origin[1] + static_cast<double>(j) * spacing[1];
        const double z = origin[2] + static_cast<double>(k) * spacing[2];
        u[idx++] = ic(x, y, z);
      }
    }
  }
}

} // namespace heat3d
