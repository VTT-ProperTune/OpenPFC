// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file brick_iteration.hpp
 * @brief Iteration helpers for `pfc::data::Field<T, MemorySpace>`.
 *
 * @details
 * Two flavours of helpers live here:
 *
 *  - **Index-aggregate form** (preferred for new code)
 *    - `for_each(field, fn)`               — every owned cell, lambda
 *                                             takes `const pfc::Int3&`.
 *    - `for_each_omp(field, fn)`           — same domain, OMP-parallel.
 *
 *    Pair these with `pfc::gradient::evaluate(grad, idx)` to keep the
 *    inner loop dimension-agnostic and free of `(i, j, k)` boilerplate.
 *
 *  - **Explicit `(int i, int j, int k)` form** (kept for laboratory-
 *    style stencils that read offsets directly):
 *    - `for_each_owned(field, fn)`         — every owned cell `[0, n)^3`.
 *    - `for_each_inner(field, r, fn)`      — owned cells whose `r`-radius
 *                                             stencil lies inside the owned
 *                                             region (`[r, n-r)^3`);
 *                                             safe to compute **before**
 *                                             halos arrive.
 *    - `for_each_coords(field, fn)`         — every **owned** cell with
 *                                             physical `(x,y,z)` and a
 *                                             mutable `T&` (or `const T&`
 *                                             on a const field).
 *    - `for_each_border(field, r, fn)`     — the rest of the owned region;
 *                                             **needs** halo data.
 *
 * Both forms iterate the same owned region and observe the same
 * **k-outer / j-middle / i-inner** order so the inner loop is cache-
 * friendly. Bodies may freely read `field(i ± r, ...)` for local access
 * patterns.
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
 * @see include/openpfc/kernel/data/field.hpp for the modern field data layout.
 */

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

// =============================================================================
// data::Field iteration helpers (M2 migration complete)
// =============================================================================

/**
 * @brief Iterate every owned cell of `field`, passing each as a
 *        `pfc::Int3{i, j, k}` to `fn`.
 *
 * This overload works with `pfc::data::Field<T, HostSpace>` and delegates
 * to the field's `for_each_owned` member function, transforming the
 * `(int,i,int j,int k)` callback into a `pfc::Int3` aggregate.
 */
template <typename T, typename MemorySpace, class Fn>
inline void for_each(const pfc::data::Field<T, MemorySpace> &field, Fn &&fn) {
  field.for_each_owned([&](int i, int j, int k) { fn(pfc::Int3{i, j, k}); });
}

/**
 * @brief OMP-parallel `for_each` for data::Field.
 *
 * Note: This serial wrapper is provided for API compatibility.
 * For true parallel iteration, use the field's native iteration methods
 * or custom OpenMP loops.
 */
template <typename T, typename MemorySpace, class Fn>
inline void for_each_omp(const pfc::data::Field<T, MemorySpace> &field, Fn &&fn) {
  // Serialize for now; Field::for_each_owned is not parallelized
  for_each(field, std::forward<Fn>(fn));
}

/**
 * @brief Iterate every owned cell (delegates to Field::for_each_owned).
 */
template <typename T, typename MemorySpace, class Fn>
inline void for_each_owned(const pfc::data::Field<T, MemorySpace> &field, Fn &&fn) {
  field.for_each_owned(std::forward<Fn>(fn));
}

/**
 * @brief OMP-parallel `for_each_owned` for data::Field (serial wrapper).
 */
template <typename T, typename MemorySpace, class Fn>
inline void for_each_owned_omp(const pfc::data::Field<T, MemorySpace> &field,
                               Fn &&fn) {
  // Serialize for now
  for_each_owned(field, std::forward<Fn>(fn));
}

/**
 * @brief Iterate the inner region `[r, nx-r) x [r, ny-r) x [r, nz-r)`.
 *
 * Delegates to the field's own iteration logic with explicit `r` parameter.
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
 * Delegates to Field::apply which already provides coordinate-based
 * iteration over every owned cell.
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
 * Border implementation for data::Field that complements `for_each_inner`.
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
