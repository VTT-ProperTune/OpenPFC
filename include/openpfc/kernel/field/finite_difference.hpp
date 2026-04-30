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
 *
 * Even-order separated Laplacians are **serial** in \f$(i_y,i_z)\f$ here; hybrid
 * OpenMP belongs in the driver (e.g.
 * `laplacian_even_order_interior_separated_xy_row` under one `omp parallel for` over
 * interior \f$(i_y,i_z)\f$).
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <openpfc/kernel/field/fd_apply.hpp>
#include <openpfc/kernel/field/fd_stencils.hpp>

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

namespace detail {

/**
 * Serial Laplacian contribution for one interior \f$(i_y,i_z)\f$ line: all \f$i_x\f$
 * in \f$[i_{\min},i_{\max})\f$. Caller may parallelize the \f$(i_y,i_z)\f$ loops
 * (e.g. `heat3d` with a single OpenMP `collapse(2)`).
 */
template <typename T>
inline void even_order_separated_xy_row_impl(
    const T *core, const std::array<const T *, 6> &face_halos, T *lap, int nx,
    int ny, int nz, int hw, int imin, int imax, int jmin, int jmax, int kmin,
    int kmax, const EvenFdStencil1d &st, T sx, T sy, T sz, int iy, int iz) {
  if (iy < jmin || iy >= jmax || iz < kmin || iz >= kmax) {
    return;
  }
  const int M = st.half_width;
  const std::int64_t *coeff = st.coeffs;
  const int sxy = nx * ny;
  const T *hpx = face_halos[0];
  const T *hnx = face_halos[1];
  const T *hpy = face_halos[2];
  const T *hny = face_halos[3];
  const T *hpz = face_halos[4];
  const T *hnz = face_halos[5];

  auto ux_at = [&](int ix, int iyl, int izl, std::size_t c, int jx) -> T {
    if (jx >= 0 && jx < nx) {
      return core[c + static_cast<std::ptrdiff_t>(jx - ix)];
    }
    if (jx < 0) {
      const std::size_t lx = static_cast<std::size_t>(jx - imin + hw);
      return hnx[static_cast<std::size_t>(izl) * static_cast<std::size_t>(ny * hw) +
                 static_cast<std::size_t>(iyl) * static_cast<std::size_t>(hw) + lx];
    }
    const std::size_t lx = static_cast<std::size_t>(jx - imax);
    return hpx[static_cast<std::size_t>(izl) * static_cast<std::size_t>(ny * hw) +
               static_cast<std::size_t>(iyl) * static_cast<std::size_t>(hw) + lx];
  };
  auto uy_at = [&](int ix, int iyl, int izl, std::size_t c, int jy) -> T {
    if (jy >= 0 && jy < ny) {
      return core[c + static_cast<std::ptrdiff_t>(jy - iyl) *
                          static_cast<std::ptrdiff_t>(nx)];
    }
    if (jy < 0) {
      const std::size_t ly = static_cast<std::size_t>(jy - jmin + hw);
      return hny[static_cast<std::size_t>(izl) * static_cast<std::size_t>(nx * hw) +
                 ly * static_cast<std::size_t>(nx) + static_cast<std::size_t>(ix)];
    }
    const std::size_t ly = static_cast<std::size_t>(jy - jmax);
    return hpy[static_cast<std::size_t>(izl) * static_cast<std::size_t>(nx * hw) +
               ly * static_cast<std::size_t>(nx) + static_cast<std::size_t>(ix)];
  };
  auto uz_at = [&](int ix, int iyl, int izl, std::size_t c, int jz) -> T {
    if (jz >= 0 && jz < nz) {
      return core[c + static_cast<std::ptrdiff_t>(jz - izl) *
                          static_cast<std::ptrdiff_t>(sxy)];
    }
    if (jz < 0) {
      const std::size_t lz = static_cast<std::size_t>(jz - kmin + hw);
      return hnz[lz * static_cast<std::size_t>(sxy) +
                 static_cast<std::size_t>(iyl) * static_cast<std::size_t>(nx) +
                 static_cast<std::size_t>(ix)];
    }
    const std::size_t lz = static_cast<std::size_t>(jz - kmax);
    return hpz[lz * static_cast<std::size_t>(sxy) +
               static_cast<std::size_t>(iyl) * static_cast<std::size_t>(nx) +
               static_cast<std::size_t>(ix)];
  };

  for (int ix = imin; ix < imax; ++ix) {
    const std::size_t c =
        static_cast<std::size_t>(ix) +
        static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
        static_cast<std::size_t>(iz) * static_cast<std::size_t>(sxy);
    const T uc = core[c];
    T dxx = static_cast<T>(coeff[0]) * uc;
    T dyy = static_cast<T>(coeff[0]) * uc;
    T dzz = static_cast<T>(coeff[0]) * uc;
    for (int k = 1; k <= M; ++k) {
      const T ck = static_cast<T>(coeff[k]);
      dxx += ck * (ux_at(ix, iy, iz, c, ix - k) + ux_at(ix, iy, iz, c, ix + k));
      dyy += ck * (uy_at(ix, iy, iz, c, iy - k) + uy_at(ix, iy, iz, c, iy + k));
      dzz += ck * (uz_at(ix, iy, iz, c, iz - k) + uz_at(ix, iy, iz, c, iz + k));
    }
    lap[c] = sx * dxx + sy * dyy + sz * dzz;
  }
}

} // namespace detail

/**
 * @brief Even-order central 3D Laplacian (\f$2\leq\text{order}\leq 20\f$, step 2)
 * with separated face halos.
 *
 * Along each axis uses the maximal-accuracy symmetric second-derivative stencil on
 * \f$(\text{order}+1)\f$ points. Requires `halo_width >= order/2` and a non-empty
 * interior slab. Invalid `order` is a no-op. **Serial** over \f$(i_y,i_z)\f$; for
 * OpenMP, call `laplacian_even_order_interior_separated_xy_row` from a driver
 * parallel loop over interior \f$(i_y,i_z)\f$.
 *
 * @see laplacian_even_order_interior_separated_xy_row
 * @note Explicit time stepping remains subject to a CFL limit.
 */
template <typename T>
void laplacian_even_order_interior_separated(
    const T *core, const std::array<const T *, 6> &face_halos, T *lap, int nx,
    int ny, int nz, T inv_dx2, T inv_dy2, T inv_dz2, int halo_width, int order) {
  detail::EvenFdStencil1d st{};
  if (!detail::fd_even_order_lookup(order, &st)) {
    return;
  }
  const int M = st.half_width;
  const std::int64_t den = st.denom;
  const int hw = halo_width;
  const int imin = hw;
  const int imax = nx - hw;
  const int jmin = hw;
  const int jmax = ny - hw;
  const int kmin = hw;
  const int kmax = nz - hw;
  if (hw < M || imin >= imax || jmin >= jmax || kmin >= kmax) {
    return;
  }

  const T inv_den = T{1} / static_cast<T>(den);
  const T sx = inv_dx2 * inv_den;
  const T sy = inv_dy2 * inv_den;
  const T sz = inv_dz2 * inv_den;

  for (int iz = kmin; iz < kmax; ++iz) {
    for (int iy = jmin; iy < jmax; ++iy) {
      detail::even_order_separated_xy_row_impl(core, face_halos, lap, nx, ny, nz, hw,
                                               imin, imax, jmin, jmax, kmin, kmax,
                                               st, sx, sy, sz, iy, iz);
    }
  }
}

/**
 * @brief One interior \f$(i_y,i_z)\f$ line of the even-order separated Laplacian.
 *
 * Computes `lap` for all interior \f$i_x\f$ at fixed \p iy, \p iz. Intended to be
 * invoked from a **single** outer OpenMP loop over \f$(i_y,i_z)\f$ in the
 * application (the kernel itself stays serial along \f$x\f$).
 */
template <typename T>
void laplacian_even_order_interior_separated_xy_row(
    const T *core, const std::array<const T *, 6> &face_halos, T *lap, int nx,
    int ny, int nz, T inv_dx2, T inv_dy2, T inv_dz2, int halo_width, int order,
    int iy, int iz) {
  detail::EvenFdStencil1d st{};
  if (!detail::fd_even_order_lookup(order, &st)) {
    return;
  }
  const int M = st.half_width;
  const std::int64_t den = st.denom;
  const int hw = halo_width;
  const int imin = hw;
  const int imax = nx - hw;
  const int jmin = hw;
  const int jmax = ny - hw;
  const int kmin = hw;
  const int kmax = nz - hw;
  if (hw < M || imin >= imax || jmin >= jmax || kmin >= kmax) {
    return;
  }
  const T inv_den = T{1} / static_cast<T>(den);
  const T sx = inv_dx2 * inv_den;
  const T sy = inv_dy2 * inv_den;
  const T sz = inv_dz2 * inv_den;
  detail::even_order_separated_xy_row_impl(core, face_halos, lap, nx, ny, nz, hw,
                                           imin, imax, jmin, jmax, kmin, kmax, st,
                                           sx, sy, sz, iy, iz);
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
  laplacian_even_order_interior_separated(core, face_halos, lap, nx, ny, nz, inv_dx2,
                                          inv_dy2, inv_dz2, halo_width, 4);
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
  laplacian_even_order_interior_separated(core, face_halos, lap, nx, ny, nz, inv_dx2,
                                          inv_dy2, inv_dz2, halo_width, 6);
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

// ============================================================================
// Templated brick Laplacians (built on `fd_apply.hpp::apply_d2_along`).
//
// All variants below take spatial accuracy as the template parameter `Order`
// (or as a runtime `int order` on the dispatcher overload). Implementations
// are one-liners over the per-axis primitive, so the inner loops fully
// unroll on the EvenCentralD2<Order> coefficient table.
//
// Two semantic flavours:
//   - "interior" (`laplacian_interior`, `laplacian2d_xy_interior`): iterate
//     over `[hw, n - hw)` along every active axis and read only from the
//     backing buffer. With `halo_width >= Order/2` every stencil arm stays
//     in-bounds, so no face halos are needed; the function works equally
//     well on an in-place layout (where boundary slabs of `buf` were
//     filled from a halo exchange) and on a separated layout (where
//     `buf == core` and the boundary slabs hold real owned values).
//   - "periodic_separated" (`laplacian_periodic_separated`,
//     `laplacian2d_xy_periodic_separated`): iterate over the full owned
//     region `[0, n)` along every active axis; for stencil arms that fall
//     outside `[0, n)`, read from the matching `face_halos` slab. This is
//     the correct full-domain primitive for periodic problems whose owned
//     region lives in a separated `core` buffer (e.g.
//     `apps/allen_cahn/`, `examples/15_finite_difference_heat.cpp`).
// ============================================================================

/**
 * @brief Even-order central 3D Laplacian on the interior `[hw, n-hw)` slab.
 *
 * Drop-in replacement for the legacy `laplacian_7point_interior` (with
 * `Order = 2`) and the all-owned half of `laplacian_even_order_interior_*`.
 * `buf` may be an in-place-laid-out array (boundary slabs already filled
 * from a halo exchange) or the `core` buffer of a separated layout
 * (boundary slabs hold real owned values); the function reads `buf` only,
 * never `face_halos`.
 *
 * @tparam Order  Even spatial order in `{2, 4, …, 20}`.
 * @param buf  Source field (`nx*ny*nz`, x fastest).
 * @param lap  Destination buffer; only interior cells are written.
 * @param halo_width  Must be `>= Order/2`; otherwise the call is a no-op.
 *
 * @note Out-of-range `Order` is a compile-time error via
 *       `EvenCentralD2<Order>`'s `static_assert`.
 */
template <int Order, class T>
void laplacian_interior(const T *buf, T *lap, int nx, int ny, int nz, T inv_dx2,
                        T inv_dy2, T inv_dz2, int halo_width) {
  using S = EvenCentralD2<Order>;
  constexpr int M = S::half_width;
  if (halo_width < M) return;
  const int imin = halo_width;
  const int imax = nx - halo_width;
  const int jmin = halo_width;
  const int jmax = ny - halo_width;
  const int kmin = halo_width;
  const int kmax = nz - halo_width;
  if (imin >= imax || jmin >= jmax || kmin >= kmax) return;

  const std::ptrdiff_t sx = 1;
  const std::ptrdiff_t sy = nx;
  const std::ptrdiff_t sz = static_cast<std::ptrdiff_t>(nx) * ny;
  const T inv_den = T{1} / static_cast<T>(S::denom);
  const T px = inv_dx2 * inv_den;
  const T py = inv_dy2 * inv_den;
  const T pz = inv_dz2 * inv_den;

  for (int iz = kmin; iz < kmax; ++iz) {
    for (int iy = jmin; iy < jmax; ++iy) {
      const std::ptrdiff_t row_base = static_cast<std::ptrdiff_t>(iy) * sy +
                                      static_cast<std::ptrdiff_t>(iz) * sz;
      for (int ix = imin; ix < imax; ++ix) {
        const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(ix) + row_base;
        lap[c] = px * apply_d2_along<0, S>(buf, c, sx, sy, sz) +
                 py * apply_d2_along<1, S>(buf, c, sx, sy, sz) +
                 pz * apply_d2_along<2, S>(buf, c, sx, sy, sz);
      }
    }
  }
}

/**
 * @brief Runtime-order dispatcher for `laplacian_interior<Order>`.
 *
 * Useful when the spatial order is selected at run time (CLI / JSON).
 * `order` outside `{2, 4, …, 20}` is a no-op (matches legacy behavior).
 */
template <class T>
void laplacian_interior(int order, const T *buf, T *lap, int nx, int ny, int nz,
                        T inv_dx2, T inv_dy2, T inv_dz2, int halo_width) {
  switch (order) {
  case 2:
    laplacian_interior<2>(buf, lap, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2,
                          halo_width);
    break;
  case 4:
    laplacian_interior<4>(buf, lap, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2,
                          halo_width);
    break;
  case 6:
    laplacian_interior<6>(buf, lap, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2,
                          halo_width);
    break;
  case 8:
    laplacian_interior<8>(buf, lap, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2,
                          halo_width);
    break;
  case 10:
    laplacian_interior<10>(buf, lap, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2,
                           halo_width);
    break;
  case 12:
    laplacian_interior<12>(buf, lap, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2,
                           halo_width);
    break;
  case 14:
    laplacian_interior<14>(buf, lap, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2,
                           halo_width);
    break;
  case 16:
    laplacian_interior<16>(buf, lap, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2,
                           halo_width);
    break;
  case 18:
    laplacian_interior<18>(buf, lap, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2,
                           halo_width);
    break;
  case 20:
    laplacian_interior<20>(buf, lap, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2,
                           halo_width);
    break;
  default: return;
  }
}

/**
 * @brief Even-order central 3D Laplacian on the **full owned domain** with
 *        face-halo lookup at the owned-region boundary.
 *
 * Iterates `[0, nx) x [0, ny) x [0, nz)` over the separated `core` buffer.
 * For each per-axis stencil arm that falls outside the owned region, reads
 * from the matching slab of `face_halos` (typically populated by
 * `pfc::SeparatedFaceHaloExchanger`):
 * face_halos[0]=+X, [1]=-X, [2]=+Y, [3]=-Y, [4]=+Z, [5]=-Z (recv buffers,
 * each of size `hw * (other two * nx)`).
 *
 * @tparam Order  Even spatial order in `{2, 4, …, 20}`.
 * @param halo_width  Must be `>= Order/2`; otherwise the call is a no-op.
 *
 * @note For interior-only use cases (no owned-region-edge updates), reach
 *       for `laplacian_interior` instead -- it skips the boundary checks
 *       in the inner loop entirely.
 */
template <int Order, class T>
void laplacian_periodic_separated(const T *core,
                                  const std::array<const T *, 6> &face_halos, T *lap,
                                  int nx, int ny, int nz, T inv_dx2, T inv_dy2,
                                  T inv_dz2, int halo_width) {
  using S = EvenCentralD2<Order>;
  constexpr int M = S::half_width;
  const int hw = halo_width;
  if (hw < M || nx <= 0 || ny <= 0 || nz <= 0) return;

  const std::ptrdiff_t sy = nx;
  const std::ptrdiff_t sz = static_cast<std::ptrdiff_t>(nx) * ny;
  const T inv_den = T{1} / static_cast<T>(S::denom);
  const T px = inv_dx2 * inv_den;
  const T py = inv_dy2 * inv_den;
  const T pz = inv_dz2 * inv_den;

  const T *hpx = face_halos[0];
  const T *hnx = face_halos[1];
  const T *hpy = face_halos[2];
  const T *hny = face_halos[3];
  const T *hpz = face_halos[4];
  const T *hnz = face_halos[5];

  const std::ptrdiff_t face_x_z_stride = static_cast<std::ptrdiff_t>(ny) * hw;
  const std::ptrdiff_t face_y_z_stride = static_cast<std::ptrdiff_t>(nx) * hw;

  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      const std::ptrdiff_t row_base = static_cast<std::ptrdiff_t>(iy) * sy +
                                      static_cast<std::ptrdiff_t>(iz) * sz;
      for (int ix = 0; ix < nx; ++ix) {
        const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(ix) + row_base;
        const T uc = core[c];
        T dxx = static_cast<T>(S::coeffs[0]) * uc;
        T dyy = static_cast<T>(S::coeffs[0]) * uc;
        T dzz = static_cast<T>(S::coeffs[0]) * uc;
        for (int k = 1; k <= M; ++k) {
          const T ck = static_cast<T>(S::coeffs[k]);
          const T uxm =
              (ix - k >= 0)
                  ? core[c - k]
                  : hnx[iz * face_x_z_stride + static_cast<std::ptrdiff_t>(iy) * hw +
                        (ix - k + hw)];
          const T uxp =
              (ix + k < nx)
                  ? core[c + k]
                  : hpx[iz * face_x_z_stride + static_cast<std::ptrdiff_t>(iy) * hw +
                        (ix + k - nx)];
          dxx += ck * (uxm + uxp);
          const T uym =
              (iy - k >= 0)
                  ? core[c - k * sy]
                  : hny[iz * face_y_z_stride +
                        static_cast<std::ptrdiff_t>(iy - k + hw) * nx + ix];
          const T uyp =
              (iy + k < ny)
                  ? core[c + k * sy]
                  : hpy[iz * face_y_z_stride +
                        static_cast<std::ptrdiff_t>(iy + k - ny) * nx + ix];
          dyy += ck * (uym + uyp);
          const T uzm = (iz - k >= 0)
                            ? core[c - k * sz]
                            : hnz[static_cast<std::ptrdiff_t>(iz - k + hw) * sz +
                                  static_cast<std::ptrdiff_t>(iy) * nx + ix];
          const T uzp = (iz + k < nz)
                            ? core[c + k * sz]
                            : hpz[static_cast<std::ptrdiff_t>(iz + k - nz) * sz +
                                  static_cast<std::ptrdiff_t>(iy) * nx + ix];
          dzz += ck * (uzm + uzp);
        }
        lap[c] = px * dxx + py * dyy + pz * dzz;
      }
    }
  }
}

/**
 * @brief Even-order central XY Laplacian (`nz == 1`) on the interior slab.
 *
 * 2D analogue of `laplacian_interior`. Drop-in replacement for the legacy
 * `laplacian_5point_xy_interior` (with `Order = 2`). Returns without
 * writing if `nz != 1`.
 */
template <int Order, class T>
void laplacian2d_xy_interior(const T *buf, T *lap, int nx, int ny, int nz, T inv_dx2,
                             T inv_dy2, int halo_width) {
  if (nz != 1) return;
  using S = EvenCentralD2<Order>;
  constexpr int M = S::half_width;
  if (halo_width < M) return;
  const int imin = halo_width;
  const int imax = nx - halo_width;
  const int jmin = halo_width;
  const int jmax = ny - halo_width;
  if (imin >= imax || jmin >= jmax) return;

  const std::ptrdiff_t sx = 1;
  const std::ptrdiff_t sy = nx;
  const std::ptrdiff_t sz = static_cast<std::ptrdiff_t>(nx) * ny;
  const T inv_den = T{1} / static_cast<T>(S::denom);
  const T px = inv_dx2 * inv_den;
  const T py = inv_dy2 * inv_den;
  constexpr int iz = 0;

  for (int iy = jmin; iy < jmax; ++iy) {
    const std::ptrdiff_t row_base =
        static_cast<std::ptrdiff_t>(iy) * sy + static_cast<std::ptrdiff_t>(iz) * sz;
    for (int ix = imin; ix < imax; ++ix) {
      const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(ix) + row_base;
      lap[c] = px * apply_d2_along<0, S>(buf, c, sx, sy, sz) +
               py * apply_d2_along<1, S>(buf, c, sx, sy, sz);
    }
  }
}

/**
 * @brief Even-order central XY Laplacian (`nz == 1`) on the **full owned
 *        domain** with face-halo lookup at the owned-region boundary.
 *
 * 2D analogue of `laplacian_periodic_separated`. Drop-in replacement for
 * the legacy `laplacian_5point_xy_periodic_separated` (with `Order = 2`).
 * Returns without writing if `nz != 1`. Uses face_halos slots 0..3
 * (+X, -X, +Y, -Y); slots 4..5 are ignored.
 */
template <int Order, class T>
void laplacian2d_xy_periodic_separated(const T *core,
                                       const std::array<const T *, 6> &face_halos,
                                       T *lap, int nx, int ny, int nz, T inv_dx2,
                                       T inv_dy2, int halo_width) {
  if (nz != 1) return;
  using S = EvenCentralD2<Order>;
  constexpr int M = S::half_width;
  const int hw = halo_width;
  if (hw < M || nx <= 0 || ny <= 0) return;

  const T inv_den = T{1} / static_cast<T>(S::denom);
  const T px = inv_dx2 * inv_den;
  const T py = inv_dy2 * inv_den;

  const T *hpx = face_halos[0];
  const T *hnx = face_halos[1];
  const T *hpy = face_halos[2];
  const T *hny = face_halos[3];

  constexpr int iz = 0;
  const std::ptrdiff_t sxy = static_cast<std::ptrdiff_t>(nx) * ny;
  const std::ptrdiff_t face_x_z_stride = static_cast<std::ptrdiff_t>(ny) * hw;
  const std::ptrdiff_t face_y_z_stride = static_cast<std::ptrdiff_t>(nx) * hw;

  for (int iy = 0; iy < ny; ++iy) {
    const std::ptrdiff_t row_base = static_cast<std::ptrdiff_t>(iy) * nx + iz * sxy;
    for (int ix = 0; ix < nx; ++ix) {
      const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(ix) + row_base;
      const T uc = core[c];
      T dxx = static_cast<T>(S::coeffs[0]) * uc;
      T dyy = static_cast<T>(S::coeffs[0]) * uc;
      for (int k = 1; k <= M; ++k) {
        const T ck = static_cast<T>(S::coeffs[k]);
        const T uxm =
            (ix - k >= 0)
                ? core[c - k]
                : hnx[iz * face_x_z_stride + static_cast<std::ptrdiff_t>(iy) * hw +
                      (ix - k + hw)];
        const T uxp =
            (ix + k < nx)
                ? core[c + k]
                : hpx[iz * face_x_z_stride + static_cast<std::ptrdiff_t>(iy) * hw +
                      (ix + k - nx)];
        dxx += ck * (uxm + uxp);
        const T uym = (iy - k >= 0)
                          ? core[c - k * nx]
                          : hny[iz * face_y_z_stride +
                                static_cast<std::ptrdiff_t>(iy - k + hw) * nx + ix];
        const T uyp = (iy + k < ny)
                          ? core[c + k * nx]
                          : hpy[iz * face_y_z_stride +
                                static_cast<std::ptrdiff_t>(iy + k - ny) * nx + ix];
        dyy += ck * (uym + uyp);
      }
      lap[c] = px * dxx + py * dyy;
    }
  }
}

} // namespace pfc::field::fd
