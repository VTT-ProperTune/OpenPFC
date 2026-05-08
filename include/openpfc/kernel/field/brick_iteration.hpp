// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file brick_iteration.hpp
 * @brief Iteration helpers for `PaddedBrick<T>`.
 *
 * @details
 * Two flavours of helpers live here:
 *
 *  - **Index-aggregate form** (preferred for new code)
 *    - `for_each(brick, fn)`                — every owned cell, lambda
 *                                             takes `const pfc::Int3&`.
 *    - `for_each_omp(brick, fn)`            — same domain, OMP-parallel.
 *
 *    Pair these with `pfc::gradient::evaluate(grad, idx)` to keep the
 *    inner loop dimension-agnostic and free of `(i, j, k)` boilerplate.
 *
 *  - **Explicit `(int i, int j, int k)` form** (kept for laboratory-
 *    style stencils that read offsets directly):
 *    - `for_each_owned(brick, fn)`          — every owned cell `[0, n)^3`.
 *    - `for_each_inner(brick, r, fn)`       — owned cells whose `r`-radius
 *                                             stencil lies inside the owned
 *                                             region (`[r, n-r)^3`);
 *                                             safe to compute **before**
 *                                             halos arrive.
 *    - `for_each_coords(brick, fn)`          — every **owned** cell with
 *                                             physical `(x,y,z)` and a
 *                                             mutable `T&` (or `const T&`
 *                                             on a const brick); halo width
 *                                             is not passed in — the brick
 *                                             already knows the layout.
 *    - `for_each_border(brick, r, fn)`      — the rest of the owned region;
 *                                             **needs** halo data.
 *
 * Both forms iterate the same owned region and observe the same
 * **k-outer / j-middle / i-inner** order so the inner loop is cache-
 * friendly. Bodies may freely read `brick(i ± r, ...)` thanks to the
 * padded layout.
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

#include <type_traits>
#include <utility>

#include <openpfc/kernel/field/padded_brick.hpp>

namespace pfc::field {

namespace detail {

template <class Fn, class T>
inline void invoke_coords_value_(Fn &&fn, double x, double y, double z, const T &v) {
  static_assert(std::is_invocable_v<Fn &, double, double, double, const T &> ||
                    std::is_invocable_v<Fn &, const pfc::Real3 &, const T &>,
                "callback must be invocable as "
                "(double x, double y, double z, const T& value) or "
                "(const Real3& xyz, const T& value)");
  if constexpr (std::is_invocable_v<Fn &, double, double, double, const T &>) {
    std::forward<Fn>(fn)(x, y, z, v);
  } else {
    std::forward<Fn>(fn)(pfc::Real3{x, y, z}, v);
  }
}

template <class Fn, class T>
inline void invoke_coords_mutable_(Fn &&fn, double x, double y, double z, T &v) {
  static_assert(std::is_invocable_v<Fn &, double, double, double, T &> ||
                    std::is_invocable_v<Fn &, const pfc::Real3 &, T &>,
                "for_each_coords: lambda must be invocable as "
                "(double x, double y, double z, T& value) or "
                "(const Real3& xyz, T& value)");
  if constexpr (std::is_invocable_v<Fn &, double, double, double, T &>) {
    std::forward<Fn>(fn)(x, y, z, v);
  } else {
    std::forward<Fn>(fn)(pfc::Real3{x, y, z}, v);
  }
}

} // namespace detail

/**
 * @brief Iterate every owned cell of `brick`, passing each as a
 *        `pfc::Int3{i, j, k}` to `fn`.
 *
 * Preferred form for code that hands the index straight to a gradient
 * evaluator or to `brick(idx)`:
 *
 * @code
 * pfc::field::for_each(du, [&](const auto& idx) {
 *   const auto g = pfc::gradient::evaluate(grad, idx);
 *   du[idx] = g.xx + g.yy + g.zz;
 * });
 * @endcode
 *
 * Iteration order is k-outer / j-middle / i-inner, matching the brick's
 * row-major (x-fastest) storage.
 *
 * Lambda signature: any callable invocable as `void(const pfc::Int3&)`
 * or `void(pfc::Int3)`.
 */
template <class T, class Fn>
inline void for_each(const PaddedBrick<T> &brick, Fn &&fn) {
  const int nx = brick.nx();
  const int ny = brick.ny();
  const int nz = brick.nz();
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        fn(pfc::Int3{i, j, k});
      }
    }
  }
}

/**
 * @brief OMP-parallel `for_each`. Identical iteration domain and order;
 *        the body must be race-free with respect to writes to shared
 *        cells (per-cell writes via `brick(idx)` are race-free).
 */
template <class T, class Fn>
inline void for_each_omp(const PaddedBrick<T> &brick, Fn &&fn) {
  const int nx = brick.nx();
  const int ny = brick.ny();
  const int nz = brick.nz();
#pragma omp parallel for collapse(2) schedule(static)
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        fn(pfc::Int3{i, j, k});
      }
    }
  }
}

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
 * @brief Iterate every **owned** cell with physical coordinates.
 *
 * Visits `(i,j,k) in [0,nx) x [0,ny) x [0,nz)` in k-outer / j-middle /
 * i-inner order. On a **non-const** `brick`, `fn` is invoked as either:
 *  - `void(double x, double y, double z, T& value)`, or
 *  - `void(const Real3& xyz, T& value)`.
 *
 * On a **const** `brick`, the value is read-only:
 *  - `void(double x, double y, double z, const T& value)`, or
 *  - `void(const Real3& xyz, const T& value)`.
 *
 * Use this for initial conditions and diagnostics that need `(x,y,z)`
 * without threading a separate `hw` argument — the brick already carries
 * decomposition, spacing, and origin. For an **interior-only** strip
 * (e.g. L2 vs an infinite-domain reference away from owned boundaries),
 * combine `brick.indices_inner(brick.halo_width())` with `global_xyz` /
 * `operator[]`.
 */
template <class T, class Fn>
inline void for_each_coords(PaddedBrick<T> &brick, Fn &&fn) {
  const int nx = brick.nx();
  const int ny = brick.ny();
  const int nz = brick.nz();
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const auto [x, y, z] = brick.global_xyz(i, j, k);
        detail::invoke_coords_mutable_(std::forward<Fn>(fn), x, y, z,
                                       brick(i, j, k));
      }
    }
  }
}

template <class T, class Fn>
inline void for_each_coords(const PaddedBrick<T> &brick, Fn &&fn) {
  const int nx = brick.nx();
  const int ny = brick.ny();
  const int nz = brick.nz();
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const auto [x, y, z] = brick.global_xyz(i, j, k);
        detail::invoke_coords_value_(std::forward<Fn>(fn), x, y, z, brick(i, j, k));
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
