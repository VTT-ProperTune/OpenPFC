// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file brick_iteration.hpp
 * @brief Iteration helpers for padded `pfc::data::Field`.
 *
 * @details
 * Two flavours of helpers live here:
 *
 *  - **Index-aggregate form** (preferred for new code)
 *    - `for_each(field, fn)`                — every owned cell, lambda
 *                                             takes `const pfc::Int3&`.
 *    - `for_each_omp(field, fn)`            — same domain, (k, j) rows
 *                                             shared among OpenMP threads
 *                                             (serial without OpenMP).
 *
 *    Pair these with `pfc::gradient::evaluate(grad, idx)` to keep the
 *    inner loop dimension-agnostic and free of `(i, j, k)` boilerplate.
 *
 *  - **Explicit `(int i, int j, int k)` form** (kept for laboratory-
 *    style stencils that read offsets directly):
 *    - `for_each_owned(field, fn)`          — every owned cell `[0, n)^3`.
 *    - `for_each_inner(field, r, fn)`       — owned cells whose `r`-radius
 *                                             stencil lies inside the owned
 *                                             region (`[r, n-r)^3`);
 *                                             safe to compute **before**
 *                                             halos arrive.
 *    - `for_each_coords(field, fn)`         — every **owned** cell with
 *                                             physical `(x,y,z)` and a
 *                                             mutable `T&` (or `const T&`
 *                                             on a const field).
 *    - `for_each_border(field, r, fn)`      — the rest of the owned region;
 *                                             **needs** halo data.
 *
 * Both forms iterate the same owned region and observe the same
 * **k-outer / j-middle / i-inner** order so the inner loop is cache-
 * friendly. Bodies may freely read `field(i ± r, ...)` on a padded Field
 * (`storage_halo >= r`).
 *
 * @note `for_each_border` produces each border cell **exactly once** even
 *       at corners (where x-, y- and z-slabs would otherwise overlap).
 *       The implementation enumerates the six face slabs in a fixed order
 *       and skips cells already covered by an earlier slab.
 *
 * @see kernel/data/grid_field.hpp for the owning data layout.
 */

#include <array>
#include <type_traits>
#include <utility>

#include <openpfc/kernel/data/grid_field.hpp>

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
 * @brief Iterate every owned cell of `field`, passing each as a
 *        `pfc::Int3{i, j, k}` to `fn`.
 *
 * This works with `pfc::data::Field<T, MemorySpace>` and delegates
 * to the field's `for_each_owned` member, transforming the
 * `(int i, int j, int k)` callback into a `pfc::Int3` aggregate.
 *
 * @code
 * pfc::field::for_each(du, [&](const auto& idx) {
 *   const auto g = pfc::gradient::evaluate(grad, idx);
 *   du[idx] = g.xx + g.yy + g.zz;
 * });
 * @endcode
 */
template <typename T, typename MemorySpace, class Fn>
inline void for_each(const pfc::data::Field<T, MemorySpace> &field, Fn &&fn) {
  field.for_each_owned([&](int i, int j, int k) { fn(pfc::Int3{i, j, k}); });
}

/**
 * @brief Call `fn(i, j, k)` once for every cell of `[0, n[0]) x [0, n[1]) x
 *        [0, n[2])`, with the `(k, j)` rows shared among OpenMP threads.
 *
 * In a translation unit compiled with OpenMP the rows are distributed by
 * `#pragma omp parallel for collapse(2) schedule(static)` (i stays the
 * contiguous inner loop); without OpenMP this is the serial k-outer,
 * j-middle, i-inner loop. `fn` must be safe to call concurrently for
 * distinct cells, and the visiting order across threads is unspecified.
 * The extents need not belong to a Field: host staging buffers of a device
 * field (initialization, say) use it directly.
 */
template <class Fn> inline void parallel_for_cells(std::array<int, 3> n, Fn &&fn) {
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) fn(i, j, k);
}

/**
 * @brief OpenMP-parallel `for_each` for data::Field: every owned cell once,
 *        rows shared among threads (see parallel_for_cells).
 */
template <typename T, typename MemorySpace, class Fn>
inline void for_each_omp(const pfc::data::Field<T, MemorySpace> &field, Fn &&fn) {
  const auto n = field.local_size();
  parallel_for_cells({n[0], n[1], n[2]},
                     [&](int i, int j, int k) { fn(pfc::Int3{i, j, k}); });
}

/**
 * @brief Iterate every owned cell (delegates to Field::for_each_owned).
 */
template <typename T, typename MemorySpace, class Fn>
inline void for_each_owned(const pfc::data::Field<T, MemorySpace> &field, Fn &&fn) {
  field.for_each_owned(std::forward<Fn>(fn));
}

/**
 * @brief OpenMP-parallel `for_each_owned` for data::Field: `fn(i, j, k)` for
 *        every owned cell once, rows shared among threads.
 */
template <typename T, typename MemorySpace, class Fn>
inline void for_each_owned_omp(const pfc::data::Field<T, MemorySpace> &field,
                               Fn &&fn) {
  const auto n = field.local_size();
  parallel_for_cells({n[0], n[1], n[2]}, fn);
}

/**
 * @brief Iterate the inner region `[r, nx-r) x [r, ny-r) x [r, nz-r)`.
 *
 * The `r`-radius stencil stays inside the owned core, so this loop is
 * safe to start **before** the halo exchange completes. No-op if any
 * axis has `n <= 2*r`.
 */
template <typename T, typename MemorySpace, class Fn>
inline void for_each_inner(const pfc::data::Field<T, MemorySpace> &field, int r,
                           Fn &&fn) {
  const auto sz = field.local_size();
  const int nx = sz[0];
  const int ny = sz[1];
  const int nz = sz[2];
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
 * @brief OMP-parallel `for_each_inner` for data::Field (serial wrapper).
 */
template <typename T, typename MemorySpace, class Fn>
inline void for_each_inner_omp(const pfc::data::Field<T, MemorySpace> &field, int r,
                               Fn &&fn) {
  // Serialize for now
  for_each_inner(field, r, std::forward<Fn>(fn));
}

/**
 * @brief Iterate every owned cell with physical coordinates (mutable).
 *
 * `fn` is invoked as either `void(double x, double y, double z, T&)` or
 * `void(const Real3& xyz, T&)`.
 */
template <typename T, typename MemorySpace, class Fn>
inline void for_each_coords(pfc::data::Field<T, MemorySpace> &field, Fn &&fn) {
  const auto sz = field.local_size();
  for (int k = 0; k < sz[2]; ++k) {
    for (int j = 0; j < sz[1]; ++j) {
      for (int i = 0; i < sz[0]; ++i) {
        const auto c = field.coords(i, j, k);
        detail::invoke_coords_mutable_(std::forward<Fn>(fn), c[0], c[1], c[2],
                                       field(i, j, k));
      }
    }
  }
}

/**
 * @brief Iterate every owned cell with physical coordinates (const).
 */
template <typename T, typename MemorySpace, class Fn>
inline void for_each_coords(const pfc::data::Field<T, MemorySpace> &field, Fn &&fn) {
  const auto sz = field.local_size();
  for (int k = 0; k < sz[2]; ++k) {
    for (int j = 0; j < sz[1]; ++j) {
      for (int i = 0; i < sz[0]; ++i) {
        const auto c = field.coords(i, j, k);
        detail::invoke_coords_value_(std::forward<Fn>(fn), c[0], c[1], c[2],
                                     field(i, j, k));
      }
    }
  }
}

/**
 * @brief Iterate the border region (owned minus interior).
 *
 * Border = owned minus inner. Complements `for_each_inner`.
 */
template <typename T, typename MemorySpace, class Fn>
inline void for_each_border(const pfc::data::Field<T, MemorySpace> &field, int r,
                            Fn &&fn) {
  const auto sz = field.local_size();
  const int nx = sz[0];
  const int ny = sz[1];
  const int nz = sz[2];

  if (nx <= 2 * r || ny <= 2 * r || nz <= 2 * r) {
    for_each_owned(field, fn);
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
