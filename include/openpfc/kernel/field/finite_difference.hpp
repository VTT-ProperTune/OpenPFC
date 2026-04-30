// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file finite_difference.hpp
 * @brief Templated brick Laplacians for distributed fields (CPU)
 *
 * @details
 * Assumes the same local layout as `halo_pattern.hpp` / `HaloExchanger`: row-major
 * `[nx, ny, nz]` with **x varying fastest** (linear index
 * `ix + iy * nx + iz * nx * ny`).
 *
 * The API is built on top of the per-axis `apply_d2_along` primitive in
 * `fd_apply.hpp` and the compile-time stencil tables in `fd_stencils.hpp`.
 * Two semantic flavours are provided, each in 3D and 2D (`nz == 1`)
 * variants:
 *
 *  - **Interior** (`laplacian_interior<Order>`,
 *    `laplacian2d_xy_interior<Order>`, runtime-order
 *    `laplacian_interior(int order, ...)` dispatcher): iterate the
 *    interior slab `[hw, n - hw)` and read only from the input buffer.
 *    Works equally well on an in-place layout (boundary slabs filled by
 *    a halo exchange) and on a separated layout's `core` buffer
 *    (boundary slabs hold real owned values); needs no face halos in
 *    the inner loop.
 *  - **Periodic-separated** (`laplacian_periodic_separated<Order>`,
 *    `laplacian2d_xy_periodic_separated<Order>`): iterate the full
 *    owned region `[0, n)` and fall back to the matching `face_halos`
 *    slab whenever a stencil arm overflows the owned region. Halo
 *    indexing matches `pfc::SeparatedFaceHaloExchanger`. This is the
 *    correct primitive for periodic problems on a separated layout
 *    (e.g. `apps/allen_cahn/`, `examples/15_finite_difference_heat.cpp`).
 *
 * @see docs/halo_exchange.md
 * @see kernel/decomposition/halo_exchange.hpp
 * @see kernel/decomposition/separated_halo_exchange.hpp
 * @see kernel/field/fd_apply.hpp
 * @see kernel/field/fd_stencils.hpp
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <openpfc/kernel/field/fd_apply.hpp>
#include <openpfc/kernel/field/fd_stencils.hpp>

namespace pfc::field::fd {

/**
 * @brief Even-order central 3D Laplacian on the interior `[hw, n-hw)` slab.
 *
 * Iterates `[hw, nx - hw) x [hw, ny - hw) x [hw, nz - hw)` and reads only
 * from `buf`; never touches `face_halos`. `buf` may be either an
 * in-place-laid-out array (boundary slabs filled by a halo exchange) or
 * the `core` buffer of a separated layout (boundary slabs hold real
 * owned values).
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
 * `order` outside `{2, 4, …, 20}` is a no-op.
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
 * 2D analogue of `laplacian_interior`. Returns without writing if
 * `nz != 1`.
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
 * 2D analogue of `laplacian_periodic_separated`. Returns without writing
 * if `nz != 1`. Uses face_halos slots 0..3 (+X, -X, +Y, -Y); slots 4..5
 * are ignored.
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
