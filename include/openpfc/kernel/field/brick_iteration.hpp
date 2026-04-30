// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file brick_iteration.hpp
 * @brief `(i, j, k)` iteration helpers for `PaddedBrick<T>`.
 *
 * @details
 * The laboratory-style FD driver wants three explicit cell ranges:
 *
 *  - `for_each_owned(brick, fn)`        — every owned cell `[0, n)^3`,
 *                                         used for IC, Euler updates, etc.
 *  - `for_each_inner(brick, r, fn)`     — owned cells whose `r`-radius
 *                                         stencil lies entirely inside the
 *                                         owned region (`[r, n-r)^3`),
 *                                         safe to compute **before** halos
 *                                         arrive.
 *  - `for_each_border(brick, r, fn)`    — the rest of the owned region:
 *                                         the union of x/y/z slabs of
 *                                         thickness `r` adjacent to the
 *                                         owned-region boundary; **needs**
 *                                         halo data.
 *
 * Each helper invokes `fn(int i, int j, int k)` with **owned-relative**
 * indices (i.e. `0 <= i < brick.nx()`, etc.), and the lambda may freely
 * read `brick(i +/- r, ...)` thanks to the padded layout.
 *
 * The OMP-parallel variants `..._omp(...)` add a single
 * `#pragma omp parallel for collapse(2) schedule(static)` over the outer
 * `(k, j)` axes — same shape as `pfc::field::for_each_interior` in
 * `kernel/simulation/for_each_interior.hpp` so the laboratory driver
 * threads the inner stencil exactly the way the compact driver does.
 *
 * @note `for_each_border` produces each border cell **exactly once** even
 *       at corners (where x-, y- and z-slabs would otherwise overlap).
 *       The implementation enumerates the six face slabs in a fixed order
 *       and skips cells already covered by an earlier slab.
 *
 * @see kernel/field/padded_brick.hpp for the owning data layout.
 */

#include <openpfc/kernel/field/padded_brick.hpp>

namespace pfc::field {

/**
 * @brief Iterate every owned cell `(i, j, k) in [0, nx) x [0, ny) x [0, nz)`.
 *
 * Lambda signature: `void(int i, int j, int k)`.
 */
template <class T, class Fn>
inline void for_each_owned(const PaddedBrick<T> &brick, Fn &&fn) {
  const int nx = brick.nx();
  const int ny = brick.ny();
  const int nz = brick.nz();
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        fn(i, j, k);
      }
    }
  }
}

/**
 * @brief OMP-parallel `for_each_owned`. Identical iteration domain;
 *        the body should not write to shared `(i, j, k)`-indexed cells
 *        except via `brick(i, j, k)` (race-free).
 */
template <class T, class Fn>
inline void for_each_owned_omp(const PaddedBrick<T> &brick, Fn &&fn) {
  const int nx = brick.nx();
  const int ny = brick.ny();
  const int nz = brick.nz();
#pragma omp parallel for collapse(2) schedule(static)
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        fn(i, j, k);
      }
    }
  }
}

/**
 * @brief Iterate the inner region `[r, nx-r) x [r, ny-r) x [r, nz-r)`.
 *
 * The `r`-radius stencil is guaranteed to stay inside the owned core,
 * so this loop is safe to start **before** the halo exchange completes.
 * No-op if any axis has `nx <= 2*r`.
 *
 * Lambda signature: `void(int i, int j, int k)`.
 */
template <class T, class Fn>
inline void for_each_inner(const PaddedBrick<T> &brick, int r, Fn &&fn) {
  const int nx = brick.nx();
  const int ny = brick.ny();
  const int nz = brick.nz();
  if (nx <= 2 * r || ny <= 2 * r || nz <= 2 * r) return;
  for (int k = r; k < nz - r; ++k) {
    for (int j = r; j < ny - r; ++j) {
      for (int i = r; i < nx - r; ++i) {
        fn(i, j, k);
      }
    }
  }
}

/**
 * @brief OMP-parallel `for_each_inner`. Same domain as the serial version.
 */
template <class T, class Fn>
inline void for_each_inner_omp(const PaddedBrick<T> &brick, int r, Fn &&fn) {
  const int nx = brick.nx();
  const int ny = brick.ny();
  const int nz = brick.nz();
  if (nx <= 2 * r || ny <= 2 * r || nz <= 2 * r) return;
#pragma omp parallel for collapse(2) schedule(static)
  for (int k = r; k < nz - r; ++k) {
    for (int j = r; j < ny - r; ++j) {
      for (int i = r; i < nx - r; ++i) {
        fn(i, j, k);
      }
    }
  }
}

/**
 * @brief Iterate the **border** region: owned cells whose `r`-radius
 *        stencil reaches into the halo.
 *
 * Border = owned region minus inner region. Concretely the union of
 * six slabs of thickness `r` on each face of the owned cube. This
 * helper visits each border cell exactly once, in the following slab
 * order:
 *
 *   1. `i in [0, r)`           (left x-slab, full y/z)
 *   2. `i in [nx-r, nx)`       (right x-slab, full y/z)
 *   3. `j in [0, r)`,          excluding cells already in slabs 1/2
 *   4. `j in [ny-r, ny)`,      excluding cells already in slabs 1/2
 *   5. `k in [0, r)`,          excluding all earlier slabs
 *   6. `k in [nz-r, nz)`,      excluding all earlier slabs
 *
 * If `nx, ny` or `nz` is `<= 2*r`, the inner region is empty and the
 * border degenerates to **every** owned cell.
 *
 * Lambda signature: `void(int i, int j, int k)`.
 */
template <class T, class Fn>
inline void for_each_border(const PaddedBrick<T> &brick, int r, Fn &&fn) {
  const int nx = brick.nx();
  const int ny = brick.ny();
  const int nz = brick.nz();

  if (nx <= 2 * r || ny <= 2 * r || nz <= 2 * r) {
    for_each_owned(brick, fn);
    return;
  }

  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < r; ++i) fn(i, j, k);
      for (int i = nx - r; i < nx; ++i) fn(i, j, k);
    }
  }

  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < r; ++j) {
      for (int i = r; i < nx - r; ++i) fn(i, j, k);
    }
    for (int j = ny - r; j < ny; ++j) {
      for (int i = r; i < nx - r; ++i) fn(i, j, k);
    }
  }

  for (int k = 0; k < r; ++k) {
    for (int j = r; j < ny - r; ++j) {
      for (int i = r; i < nx - r; ++i) fn(i, j, k);
    }
  }
  for (int k = nz - r; k < nz; ++k) {
    for (int j = r; j < ny - r; ++j) {
      for (int i = r; i < nx - r; ++i) fn(i, j, k);
    }
  }
}

} // namespace pfc::field
