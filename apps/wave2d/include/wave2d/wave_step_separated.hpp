// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file wave_step_separated.hpp
 * @brief One explicit Euler step for (u,v) on separated halos (order-2 XY
 *        Laplacian), with optional physical y-BC patch on face buffers after
 *        the periodic exchange. Used by GPU parity tests and CUDA/HIP drivers.
 */

#include <array>
#include <cstddef>
#include <vector>

#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/separated_halo_exchange.hpp>
#include <openpfc/kernel/field/finite_difference.hpp>

#include <wave2d/wave_boundary.hpp>
#include <wave2d/wave_model.hpp>

namespace wave2d {

/** Patch -Y / +Y face recv buffers for global y Dirichlet walls (order-2). */
inline void patch_y_face_halos_dirichlet_order2(
    const double *u_core, int nx, int ny, std::array<std::vector<double>, 6> &faces,
    const pfc::Int3 &lower_global, int Ny_global, double u_wall) {
  constexpr int hw = 1;
  if (ny < 1 || nx < 1) return;
  const int iz = 0;
  const std::ptrdiff_t face_y_z_stride = static_cast<std::ptrdiff_t>(nx) * hw;

  if (lower_global[1] == 0) {
    double *hny = faces[static_cast<std::size_t>(3)].data();
    for (int ix = 0; ix < nx; ++ix) {
      const std::size_t c = static_cast<std::size_t>(ix) +
                            static_cast<std::size_t>(iz) * face_y_z_stride;
      const std::size_t uc =
          static_cast<std::size_t>(ix) +
          static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx * ny);
      hny[c] = 2.0 * u_wall - u_core[uc];
    }
  }
  if (lower_global[1] + ny - 1 == Ny_global - 1) {
    double *hpy = faces[static_cast<std::size_t>(2)].data();
    for (int ix = 0; ix < nx; ++ix) {
      const std::size_t c = static_cast<std::size_t>(ix) +
                            static_cast<std::size_t>(iz) * face_y_z_stride;
      const std::size_t uc =
          static_cast<std::size_t>(ix) +
          static_cast<std::size_t>(ny - 1) * static_cast<std::size_t>(nx) +
          static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx * ny);
      hpy[c] = 2.0 * u_wall - u_core[uc];
    }
  }
}

/** Homogeneous Neumann: mirror u into -Y / +Y face buffers. */
inline void
patch_y_face_halos_neumann_order2(const double *u_core, int nx, int ny,
                                  std::array<std::vector<double>, 6> &faces,
                                  const pfc::Int3 &lower_global, int Ny_global) {
  constexpr int hw = 1;
  if (ny < 2 || nx < 1) return;
  const int iz = 0;
  const std::ptrdiff_t face_y_z_stride = static_cast<std::ptrdiff_t>(nx) * hw;

  if (lower_global[1] == 0) {
    double *hny = faces[static_cast<std::size_t>(3)].data();
    for (int ix = 0; ix < nx; ++ix) {
      const std::size_t c = static_cast<std::size_t>(ix) +
                            static_cast<std::size_t>(iz) * face_y_z_stride;
      const std::size_t uc1 =
          static_cast<std::size_t>(ix) +
          static_cast<std::size_t>(1) * static_cast<std::size_t>(nx) +
          static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx * ny);
      hny[c] = u_core[uc1];
    }
  }
  if (lower_global[1] + ny - 1 == Ny_global - 1) {
    double *hpy = faces[static_cast<std::size_t>(2)].data();
    for (int ix = 0; ix < nx; ++ix) {
      const std::size_t c = static_cast<std::size_t>(ix) +
                            static_cast<std::size_t>(iz) * face_y_z_stride;
      const std::size_t ucm =
          static_cast<std::size_t>(ix) +
          static_cast<std::size_t>(ny - 2) * static_cast<std::size_t>(nx) +
          static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx * ny);
      hpy[c] = u_core[ucm];
    }
  }
}

inline void step_wave_separated_order2_cpu(
    std::vector<double> &u, std::vector<double> &v, std::vector<double> &lap,
    std::array<std::vector<double>, 6> &face_halos,
    pfc::SeparatedFaceHaloExchanger<double> &exchanger, int nx, int ny, int nz,
    const pfc::decomposition::Decomposition &decomp, int rank, double dt,
    YBoundaryKind y_bc, int Ny_global, double u_wall) {
  exchanger.exchange_halos(u.data(), u.size(), face_halos);

  const auto &local = pfc::decomposition::get_subworld(decomp, rank);
  const auto lower = pfc::world::get_lower(local);

  if (y_bc == YBoundaryKind::Dirichlet) {
    patch_y_face_halos_dirichlet_order2(u.data(), nx, ny, face_halos, lower,
                                        Ny_global, u_wall);
  } else {
    patch_y_face_halos_neumann_order2(u.data(), nx, ny, face_halos, lower,
                                      Ny_global);
  }

  std::array<const double *, 6> face_ptrs{};
  for (int i = 0; i < 6; ++i) {
    face_ptrs[static_cast<std::size_t>(i)] =
        face_halos[static_cast<std::size_t>(i)].data();
  }

  const double inv_h2 = 1.0;
  pfc::field::fd::laplacian2d_xy_periodic_separated<2>(
      u.data(), face_ptrs, lap.data(), nx, ny, nz, inv_h2, inv_h2, 1);

  const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                        static_cast<std::size_t>(nz);
  for (std::size_t i = 0; i < n; ++i) {
    const double vc = v[i];
    const double lc = lap[i];
    u[i] += dt * vc;
    v[i] += dt * kC * kC * lc;
  }

  if (y_bc == YBoundaryKind::Dirichlet) {
    for (int iz = 0; iz < nz; ++iz) {
      for (int iy = 0; iy < ny; ++iy) {
        const int gy = lower[1] + iy;
        if (gy != 0 && gy != Ny_global - 1) {
          continue;
        }
        for (int ix = 0; ix < nx; ++ix) {
          const std::size_t idx =
              static_cast<std::size_t>(ix) +
              static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
              static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx * ny);
          u[idx] = u_wall;
          v[idx] = 0.0;
        }
      }
    }
  }
}

} // namespace wave2d
