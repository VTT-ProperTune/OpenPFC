// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file finite_difference.hpp
 * @brief Minimal real-space stencil helpers for distributed fields (CPU)
 *
 * @details
 * Assumes the same local layout as `halo_pattern.hpp` / `HaloExchanger`: row-major
 * `[nx, ny, nz]` with **x varying fastest** (linear index
 * `ix + iy * nx + iz * nx * ny`).
 *
 * After a halo exchange with `halo_width` ghost layers, apply stencils only on the
 * interior slab where all neighbors lie in the local array (see
 * `laplacian_7point_interior`).
 *
 * @see docs/halo_exchange.md
 * @see kernel/decomposition/halo_exchange.hpp
 * @see kernel/decomposition/separated_halo_exchange.hpp
 */

#pragma once

#include <array>
#include <cstddef>

namespace pfc::field::fd {

/**
 * @brief 7-point Laplacian on interior indices (periodic ghost layout).
 *
 * Writes `lap[out]` for each interior point; entries outside the interior range are
 * untouched (caller may zero `lap` first).
 *
 * @param u Source field (same layout as after halo exchange)
 * @param lap Output buffer (same size as u)
 * @param nx, ny, nz Local field dimensions
 * @param inv_dx2, inv_dy2, inv_dz2 1/dx², 1/dy², 1/dz² for uniform grid
 * @param halo_width Ghost width from halo exchange (1 for a 3-point axis stencil)
 */
template <typename T>
void laplacian_7point_interior(const T *u, T *lap, int nx, int ny, int nz, T inv_dx2,
                               T inv_dy2, T inv_dz2, int halo_width) {
  const int imin = halo_width;
  const int imax = nx - halo_width;
  const int jmin = halo_width;
  const int jmax = ny - halo_width;
  const int kmin = halo_width;
  const int kmax = nz - halo_width;
  if (imin >= imax || jmin >= jmax || kmin >= kmax) {
    return;
  }

  const int sxy = nx * ny;
  for (int iz = kmin; iz < kmax; ++iz) {
    for (int iy = jmin; iy < jmax; ++iy) {
      for (int ix = imin; ix < imax; ++ix) {
        const std::size_t c =
            static_cast<std::size_t>(ix) +
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(iz) * static_cast<std::size_t>(sxy);
        const T uc = u[c];
        const T dxx = u[c + 1] + u[c - 1] - T{2} * uc;
        const T dyy = u[c + static_cast<std::size_t>(nx)] +
                      u[c - static_cast<std::size_t>(nx)] - T{2} * uc;
        const T dzz = u[c + static_cast<std::size_t>(sxy)] +
                      u[c - static_cast<std::size_t>(sxy)] - T{2} * uc;
        lap[c] = inv_dx2 * dxx + inv_dy2 * dyy + inv_dz2 * dzz;
      }
    }
  }
}

/**
 * @brief 7-point Laplacian with **separated** face halos (FFT-safe core).
 *
 * `core` is row-major `[nx,ny,nz]` (owned subdomain only). `face_halos[k]` points
 * to contiguous data for direction k, same order as `SeparatedFaceHaloExchanger` /
 * `halo_face_layout.hpp`:
 * 0:+X recv (low-x ghost), 1:-X recv (high-x), 2:+Y (low-y), 3:-Y (high-y),
 * 4:+Z (low-z), 5:-Z (high-z).
 *
 * Layout within each face matches `create_recv_halo` traversal (z, then y, then x
 * in global index space): e.g. for slot 0 size `hw*ny*nz`, offset
 * `iz*(ny*hw) + iy*hw + lx` with `lx` in `[0,hw)`.
 *
 * Writes `lap` at interior linear indices only (same interior slab as
 * `laplacian_7point_interior`).
 */
template <typename T>
void laplacian_7point_interior_separated(const T *core,
                                         const std::array<const T *, 6> &face_halos,
                                         T *lap, int nx, int ny, int nz, T inv_dx2,
                                         T inv_dy2, T inv_dz2, int halo_width) {
  const int hw = halo_width;
  const int imin = hw;
  const int imax = nx - hw;
  const int jmin = hw;
  const int jmax = ny - hw;
  const int kmin = hw;
  const int kmax = nz - hw;
  if (imin >= imax || jmin >= jmax || kmin >= kmax) {
    return;
  }

  const int sxy = nx * ny;
  const T *hpx = face_halos[0];
  const T *hnx = face_halos[1];
  const T *hpy = face_halos[2];
  const T *hny = face_halos[3];
  const T *hpz = face_halos[4];
  const T *hnz = face_halos[5];

  for (int iz = kmin; iz < kmax; ++iz) {
    for (int iy = jmin; iy < jmax; ++iy) {
      for (int ix = imin; ix < imax; ++ix) {
        const std::size_t c =
            static_cast<std::size_t>(ix) +
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(iz) * static_cast<std::size_t>(sxy);
        const T uc = core[c];

        const T uxm =
            (ix > imin)
                ? core[c - 1]
                : hnx[static_cast<std::size_t>(iz) *
                          static_cast<std::size_t>(ny * hw) +
                      static_cast<std::size_t>(iy) * static_cast<std::size_t>(hw) +
                      static_cast<std::size_t>(hw - 1)];
        const T uxp =
            (ix + 1 < imax)
                ? core[c + 1]
                : hpx[static_cast<std::size_t>(iz) *
                          static_cast<std::size_t>(ny * hw) +
                      static_cast<std::size_t>(iy) * static_cast<std::size_t>(hw) +
                      static_cast<std::size_t>(ix + 1 - (nx - hw))];

        const T uym = (iy > jmin) ? core[c - static_cast<std::size_t>(nx)]
                                  : hny[static_cast<std::size_t>(iz) *
                                            static_cast<std::size_t>(nx * hw) +
                                        static_cast<std::size_t>(hw - 1) *
                                            static_cast<std::size_t>(nx) +
                                        static_cast<std::size_t>(ix)];
        const T uyp = (iy + 1 < jmax)
                          ? core[c + static_cast<std::size_t>(nx)]
                          : hpy[static_cast<std::size_t>(iz) *
                                    static_cast<std::size_t>(nx * hw) +
                                static_cast<std::size_t>(iy + 1 - (ny - hw)) *
                                    static_cast<std::size_t>(nx) +
                                static_cast<std::size_t>(ix)];

        const T uzm =
            (iz > kmin)
                ? core[c - static_cast<std::size_t>(sxy)]
                : hnz[static_cast<std::size_t>(hw - 1) *
                          static_cast<std::size_t>(sxy) +
                      static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
                      static_cast<std::size_t>(ix)];
        const T uzp =
            (iz + 1 < kmax)
                ? core[c + static_cast<std::size_t>(sxy)]
                : hpz[static_cast<std::size_t>(iz + 1 - (nz - hw)) *
                          static_cast<std::size_t>(sxy) +
                      static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
                      static_cast<std::size_t>(ix)];

        const T dxx = uxp + uxm - T{2} * uc;
        const T dyy = uyp + uym - T{2} * uc;
        const T dzz = uzp + uzm - T{2} * uc;
        lap[c] = inv_dx2 * dxx + inv_dy2 * dyy + inv_dz2 * dzz;
      }
    }
  }
}

/**
 * @brief 4th-order accurate 3D Laplacian (5-point per axis) with separated face
 * halos.
 *
 * Uses the standard central second-derivative stencil along each axis:
 * \f$\partial^2 u/\partial x^2 \approx
 * (-u_{i-2}+16u_{i-1}-30u_i+16u_{i+1}-u_{i+2})/(12\,\Delta x^2)\f$. Requires
 * `halo_width >= 2` on each face. The interior slab is the same as for
 * `laplacian_7point_interior_separated` (`ix,iy,iz` in `[hw,nx-hw)` etc.).
 *
 * @note Explicit time stepping remains subject to a CFL limit; higher spatial
 *       order does not remove the stability restriction on `dt`.
 */
template <typename T>
void laplacian_4th_order_interior_separated(
    const T *core, const std::array<const T *, 6> &face_halos, T *lap, int nx,
    int ny, int nz, T inv_dx2, T inv_dy2, T inv_dz2, int halo_width) {
  const int hw = halo_width;
  const int imin = hw;
  const int imax = nx - hw;
  const int jmin = hw;
  const int jmax = ny - hw;
  const int kmin = hw;
  const int kmax = nz - hw;
  if (hw < 2 || imin >= imax || jmin >= jmax || kmin >= kmax) {
    return;
  }

  const int sxy = nx * ny;
  const T *hpx = face_halos[0];
  const T *hnx = face_halos[1];
  const T *hpy = face_halos[2];
  const T *hny = face_halos[3];
  const T *hpz = face_halos[4];
  const T *hnz = face_halos[5];

  const T c4 = inv_dx2 / T{12};
  const T c4y = inv_dy2 / T{12};
  const T c4z = inv_dz2 / T{12};

  // Neighbor column/plane j* may lie in owned core [0,n*) or in separated recv
  // buffers when j* < 0 or j* >= n* (MPI / periodic ghost outside local core).
  auto ux_at = [&](int ix, int iy, int iz, std::size_t c, int jx) -> T {
    if (jx >= 0 && jx < nx) {
      return core[c + static_cast<std::ptrdiff_t>(jx - ix)];
    }
    if (jx < 0) {
      const std::size_t lx = static_cast<std::size_t>(jx - imin + hw);
      return hnx[static_cast<std::size_t>(iz) * static_cast<std::size_t>(ny * hw) +
                 static_cast<std::size_t>(iy) * static_cast<std::size_t>(hw) + lx];
    }
    const std::size_t lx = static_cast<std::size_t>(jx - imax);
    return hpx[static_cast<std::size_t>(iz) * static_cast<std::size_t>(ny * hw) +
               static_cast<std::size_t>(iy) * static_cast<std::size_t>(hw) + lx];
  };
  auto uy_at = [&](int ix, int iy, int iz, std::size_t c, int jy) -> T {
    if (jy >= 0 && jy < ny) {
      return core[c + static_cast<std::ptrdiff_t>(jy - iy) *
                          static_cast<std::ptrdiff_t>(nx)];
    }
    if (jy < 0) {
      const std::size_t ly = static_cast<std::size_t>(jy - jmin + hw);
      return hny[static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx * hw) +
                 ly * static_cast<std::size_t>(nx) + static_cast<std::size_t>(ix)];
    }
    const std::size_t ly = static_cast<std::size_t>(jy - jmax);
    return hpy[static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx * hw) +
               ly * static_cast<std::size_t>(nx) + static_cast<std::size_t>(ix)];
  };
  auto uz_at = [&](int ix, int iy, int iz, std::size_t c, int jz) -> T {
    if (jz >= 0 && jz < nz) {
      return core[c + static_cast<std::ptrdiff_t>(jz - iz) *
                          static_cast<std::ptrdiff_t>(sxy)];
    }
    if (jz < 0) {
      const std::size_t lz = static_cast<std::size_t>(jz - kmin + hw);
      return hnz[lz * static_cast<std::size_t>(sxy) +
                 static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
                 static_cast<std::size_t>(ix)];
    }
    const std::size_t lz = static_cast<std::size_t>(jz - kmax);
    return hpz[lz * static_cast<std::size_t>(sxy) +
               static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
               static_cast<std::size_t>(ix)];
  };

  for (int iz = kmin; iz < kmax; ++iz) {
    for (int iy = jmin; iy < jmax; ++iy) {
      for (int ix = imin; ix < imax; ++ix) {
        const std::size_t c =
            static_cast<std::size_t>(ix) +
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(iz) * static_cast<std::size_t>(sxy);
        const T uc = core[c];
        const T uxm2 = ux_at(ix, iy, iz, c, ix - 2);
        const T uxm1 = ux_at(ix, iy, iz, c, ix - 1);
        const T uxp1 = ux_at(ix, iy, iz, c, ix + 1);
        const T uxp2 = ux_at(ix, iy, iz, c, ix + 2);
        const T dxx = -uxm2 + T{16} * uxm1 - T{30} * uc + T{16} * uxp1 - uxp2;

        const T uym2 = uy_at(ix, iy, iz, c, iy - 2);
        const T uym1 = uy_at(ix, iy, iz, c, iy - 1);
        const T uyp1 = uy_at(ix, iy, iz, c, iy + 1);
        const T uyp2 = uy_at(ix, iy, iz, c, iy + 2);
        const T dyy = -uym2 + T{16} * uym1 - T{30} * uc + T{16} * uyp1 - uyp2;

        const T uzm2 = uz_at(ix, iy, iz, c, iz - 2);
        const T uzm1 = uz_at(ix, iy, iz, c, iz - 1);
        const T uzp1 = uz_at(ix, iy, iz, c, iz + 1);
        const T uzp2 = uz_at(ix, iy, iz, c, iz + 2);
        const T dzz = -uzm2 + T{16} * uzm1 - T{30} * uc + T{16} * uzp1 - uzp2;

        lap[c] = c4 * dxx + c4y * dyy + c4z * dzz;
      }
    }
  }
}

/**
 * @brief 6th-order accurate 3D Laplacian (7-point per axis) with separated face
 * halos.
 *
 * Uses the standard central second-derivative stencil along each axis (7-point).
 * Requires `halo_width >= 3`. Interior slab matches
 * `laplacian_7point_interior_separated`.
 *
 * @note Explicit time stepping remains subject to a CFL limit.
 */
template <typename T>
void laplacian_6th_order_interior_separated(
    const T *core, const std::array<const T *, 6> &face_halos, T *lap, int nx,
    int ny, int nz, T inv_dx2, T inv_dy2, T inv_dz2, int halo_width) {
  const int hw = halo_width;
  const int imin = hw;
  const int imax = nx - hw;
  const int jmin = hw;
  const int jmax = ny - hw;
  const int kmin = hw;
  const int kmax = nz - hw;
  if (hw < 3 || imin >= imax || jmin >= jmax || kmin >= kmax) {
    return;
  }

  const int sxy = nx * ny;
  const T *hpx = face_halos[0];
  const T *hnx = face_halos[1];
  const T *hpy = face_halos[2];
  const T *hny = face_halos[3];
  const T *hpz = face_halos[4];
  const T *hnz = face_halos[5];

  const T c6 = inv_dx2 / T{180};
  const T c6y = inv_dy2 / T{180};
  const T c6z = inv_dz2 / T{180};

  auto ux_at = [&](int ix, int iy, int iz, std::size_t c, int jx) -> T {
    if (jx >= 0 && jx < nx) {
      return core[c + static_cast<std::ptrdiff_t>(jx - ix)];
    }
    if (jx < 0) {
      const std::size_t lx = static_cast<std::size_t>(jx - imin + hw);
      return hnx[static_cast<std::size_t>(iz) * static_cast<std::size_t>(ny * hw) +
                 static_cast<std::size_t>(iy) * static_cast<std::size_t>(hw) + lx];
    }
    const std::size_t lx = static_cast<std::size_t>(jx - imax);
    return hpx[static_cast<std::size_t>(iz) * static_cast<std::size_t>(ny * hw) +
               static_cast<std::size_t>(iy) * static_cast<std::size_t>(hw) + lx];
  };
  auto uy_at = [&](int ix, int iy, int iz, std::size_t c, int jy) -> T {
    if (jy >= 0 && jy < ny) {
      return core[c + static_cast<std::ptrdiff_t>(jy - iy) *
                          static_cast<std::ptrdiff_t>(nx)];
    }
    if (jy < 0) {
      const std::size_t ly = static_cast<std::size_t>(jy - jmin + hw);
      return hny[static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx * hw) +
                 ly * static_cast<std::size_t>(nx) + static_cast<std::size_t>(ix)];
    }
    const std::size_t ly = static_cast<std::size_t>(jy - jmax);
    return hpy[static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx * hw) +
               ly * static_cast<std::size_t>(nx) + static_cast<std::size_t>(ix)];
  };
  auto uz_at = [&](int ix, int iy, int iz, std::size_t c, int jz) -> T {
    if (jz >= 0 && jz < nz) {
      return core[c + static_cast<std::ptrdiff_t>(jz - iz) *
                          static_cast<std::ptrdiff_t>(sxy)];
    }
    if (jz < 0) {
      const std::size_t lz = static_cast<std::size_t>(jz - kmin + hw);
      return hnz[lz * static_cast<std::size_t>(sxy) +
                 static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
                 static_cast<std::size_t>(ix)];
    }
    const std::size_t lz = static_cast<std::size_t>(jz - kmax);
    return hpz[lz * static_cast<std::size_t>(sxy) +
               static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
               static_cast<std::size_t>(ix)];
  };

  for (int iz = kmin; iz < kmax; ++iz) {
    for (int iy = jmin; iy < jmax; ++iy) {
      for (int ix = imin; ix < imax; ++ix) {
        const std::size_t c =
            static_cast<std::size_t>(ix) +
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(iz) * static_cast<std::size_t>(sxy);
        const T uc = core[c];
        const T uxm3 = ux_at(ix, iy, iz, c, ix - 3);
        const T uxm2 = ux_at(ix, iy, iz, c, ix - 2);
        const T uxm1 = ux_at(ix, iy, iz, c, ix - 1);
        const T uxp1 = ux_at(ix, iy, iz, c, ix + 1);
        const T uxp2 = ux_at(ix, iy, iz, c, ix + 2);
        const T uxp3 = ux_at(ix, iy, iz, c, ix + 3);
        const T dxx = T{2} * (uxm3 + uxp3) - T{27} * (uxm2 + uxp2) +
                      T{270} * (uxm1 + uxp1) - T{490} * uc;

        const T uym3 = uy_at(ix, iy, iz, c, iy - 3);
        const T uym2 = uy_at(ix, iy, iz, c, iy - 2);
        const T uym1 = uy_at(ix, iy, iz, c, iy - 1);
        const T uyp1 = uy_at(ix, iy, iz, c, iy + 1);
        const T uyp2 = uy_at(ix, iy, iz, c, iy + 2);
        const T uyp3 = uy_at(ix, iy, iz, c, iy + 3);
        const T dyy = T{2} * (uym3 + uyp3) - T{27} * (uym2 + uyp2) +
                      T{270} * (uym1 + uyp1) - T{490} * uc;

        const T uzm3 = uz_at(ix, iy, iz, c, iz - 3);
        const T uzm2 = uz_at(ix, iy, iz, c, iz - 2);
        const T uzm1 = uz_at(ix, iy, iz, c, iz - 1);
        const T uzp1 = uz_at(ix, iy, iz, c, iz + 1);
        const T uzp2 = uz_at(ix, iy, iz, c, iz + 2);
        const T uzp3 = uz_at(ix, iy, iz, c, iz + 3);
        const T dzz = T{2} * (uzm3 + uzp3) - T{27} * (uzm2 + uzp2) +
                      T{270} * (uzm1 + uzp1) - T{490} * uc;

        lap[c] = c6 * dxx + c6y * dyy + c6z * dzz;
      }
    }
  }
}

/**
 * @brief 5-point XY Laplacian on interior indices (in-place halo layout), nz = 1.
 *
 * For a single z-layer local grid (`nz == 1`), applies only ∂²/∂x² + ∂²/∂y² using
 * neighbors in the same layer after halo exchange. If `nz != 1`, returns without
 * writing (caller should use the 7-point Laplacian for 3D slabs).
 */
template <typename T>
void laplacian_5point_xy_interior(const T *u, T *lap, int nx, int ny, int nz,
                                  T inv_dx2, T inv_dy2, int halo_width) {
  if (nz != 1) {
    return;
  }
  const int imin = halo_width;
  const int imax = nx - halo_width;
  const int jmin = halo_width;
  const int jmax = ny - halo_width;
  if (imin >= imax || jmin >= jmax) {
    return;
  }

  constexpr int iz = 0;
  const int sxy = nx * ny;
  for (int iy = jmin; iy < jmax; ++iy) {
    for (int ix = imin; ix < imax; ++ix) {
      const std::size_t c =
          static_cast<std::size_t>(ix) +
          static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
          static_cast<std::size_t>(iz) * static_cast<std::size_t>(sxy);
      const T uc = u[c];
      const T dxx = u[c + 1] + u[c - 1] - T{2} * uc;
      const T dyy = u[c + static_cast<std::size_t>(nx)] +
                    u[c - static_cast<std::size_t>(nx)] - T{2} * uc;
      lap[c] = inv_dx2 * dxx + inv_dy2 * dyy;
    }
  }
}

/**
 * @brief 5-point XY Laplacian with separated face halos, nz = 1.
 *
 * Uses face buffer slots 0–3 only (+X, −X, +Y, −Y recv buffers). Slots 4–5 may
 * still be filled by `SeparatedFaceHaloExchanger` but are ignored here.
 */
template <typename T>
void laplacian_5point_xy_interior_separated(
    const T *core, const std::array<const T *, 6> &face_halos, T *lap, int nx,
    int ny, int nz, T inv_dx2, T inv_dy2, int halo_width) {
  if (nz != 1) {
    return;
  }
  const int hw = halo_width;
  const int imin = hw;
  const int imax = nx - hw;
  const int jmin = hw;
  const int jmax = ny - hw;
  if (imin >= imax || jmin >= jmax) {
    return;
  }

  constexpr int iz = 0;
  const int sxy = nx * ny;
  const T *hpx = face_halos[0];
  const T *hnx = face_halos[1];
  const T *hpy = face_halos[2];
  const T *hny = face_halos[3];

  for (int iy = jmin; iy < jmax; ++iy) {
    for (int ix = imin; ix < imax; ++ix) {
      const std::size_t c =
          static_cast<std::size_t>(ix) +
          static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
          static_cast<std::size_t>(iz) * static_cast<std::size_t>(sxy);
      const T uc = core[c];

      const T uxm =
          (ix > imin)
              ? core[c - 1]
              : hnx[static_cast<std::size_t>(iz) *
                        static_cast<std::size_t>(ny * hw) +
                    static_cast<std::size_t>(iy) * static_cast<std::size_t>(hw) +
                    static_cast<std::size_t>(hw - 1)];
      const T uxp =
          (ix + 1 < imax)
              ? core[c + 1]
              : hpx[static_cast<std::size_t>(iz) *
                        static_cast<std::size_t>(ny * hw) +
                    static_cast<std::size_t>(iy) * static_cast<std::size_t>(hw) +
                    static_cast<std::size_t>(ix + 1 - (nx - hw))];

      const T uym =
          (iy > jmin)
              ? core[c - static_cast<std::size_t>(nx)]
              : hny[static_cast<std::size_t>(iz) *
                        static_cast<std::size_t>(nx * hw) +
                    static_cast<std::size_t>(hw - 1) * static_cast<std::size_t>(nx) +
                    static_cast<std::size_t>(ix)];
      const T uyp = (iy + 1 < jmax)
                        ? core[c + static_cast<std::size_t>(nx)]
                        : hpy[static_cast<std::size_t>(iz) *
                                  static_cast<std::size_t>(nx * hw) +
                              static_cast<std::size_t>(iy + 1 - (ny - hw)) *
                                  static_cast<std::size_t>(nx) +
                              static_cast<std::size_t>(ix)];

      const T dxx = uxp + uxm - T{2} * uc;
      const T dyy = uyp + uym - T{2} * uc;
      lap[c] = inv_dx2 * dxx + inv_dy2 * dyy;
    }
  }
}

/**
 * @brief 5-point XY Laplacian on every owned cell using separated periodic halos.
 *
 * Unlike `laplacian_5point_xy_interior_separated`, this treats `core` as the
 * owned local domain without ghost padding. Boundary-owned cells use the
 * separated face buffers for their missing ±X/±Y neighbors, so the result is
 * independent of the MPI decomposition for periodic domains.
 */
template <typename T>
void laplacian_5point_xy_periodic_separated(
    const T *core, const std::array<const T *, 6> &face_halos, T *lap, int nx,
    int ny, int nz, T inv_dx2, T inv_dy2, int halo_width) {
  if (nz != 1 || nx <= 0 || ny <= 0 || halo_width < 1) {
    return;
  }

  constexpr int iz = 0;
  const int sxy = nx * ny;
  const int hw = halo_width;
  const T *hpx = face_halos[0]; // values from the +X neighbor
  const T *hnx = face_halos[1]; // values from the -X neighbor
  const T *hpy = face_halos[2]; // values from the +Y neighbor
  const T *hny = face_halos[3]; // values from the -Y neighbor

  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      const std::size_t c =
          static_cast<std::size_t>(ix) +
          static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
          static_cast<std::size_t>(iz) * static_cast<std::size_t>(sxy);
      const T uc = core[c];

      const T uxm =
          (ix > 0)
              ? core[c - 1]
              : hnx[static_cast<std::size_t>(iz) *
                        static_cast<std::size_t>(ny * hw) +
                    static_cast<std::size_t>(iy) * static_cast<std::size_t>(hw) +
                    static_cast<std::size_t>(hw - 1)];
      const T uxp =
          (ix + 1 < nx)
              ? core[c + 1]
              : hpx[static_cast<std::size_t>(iz) *
                        static_cast<std::size_t>(ny * hw) +
                    static_cast<std::size_t>(iy) * static_cast<std::size_t>(hw)];

      const T uym =
          (iy > 0)
              ? core[c - static_cast<std::size_t>(nx)]
              : hny[static_cast<std::size_t>(iz) *
                        static_cast<std::size_t>(nx * hw) +
                    static_cast<std::size_t>(hw - 1) * static_cast<std::size_t>(nx) +
                    static_cast<std::size_t>(ix)];
      const T uyp = (iy + 1 < ny) ? core[c + static_cast<std::size_t>(nx)]
                                  : hpy[static_cast<std::size_t>(iz) *
                                            static_cast<std::size_t>(nx * hw) +
                                        static_cast<std::size_t>(ix)];

      const T dxx = uxp + uxm - T{2} * uc;
      const T dyy = uyp + uym - T{2} * uc;
      lap[c] = inv_dx2 * dxx + inv_dy2 * dyy;
    }
  }
}

} // namespace pfc::field::fd
