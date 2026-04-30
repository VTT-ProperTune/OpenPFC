// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file fd_apply.hpp
 * @brief Per-point apply primitives for central FD stencils.
 *
 * @details
 * Applies the central second-derivative stencils tabulated in
 * `fd_stencils.hpp` to a single interior point of a row-major 3D field
 * (`[nx, ny, nz]`, x varies fastest). Two flavours are exposed:
 *
 * - **Compile-time stencil**: `apply_d2_along<Axis, Stencil>(core, c,
 *   sx, sy, sz)`. With `Stencil = EvenCentralD2<Order>` the coefficient
 *   loop unrolls (`Stencil::half_width` and `Stencil::coeffs` are
 *   `static constexpr`), and the integer weights become immediates.
 *
 * - **Runtime stencil**: `apply_d2_along<Axis>(view, core, c, sx, sy,
 *   sz)`. Same arithmetic, but `view` is the runtime
 *   `EvenCentralD2View` populated by `lookup_even_central_d2(order, ...)`.
 *   Intended for callers that pick spatial order from a CLI / JSON
 *   configuration (`apps/heat3d`).
 *
 * Both forms return the **unscaled** finite-difference sum: the second
 * derivative along the chosen axis is `result / (h^2 * Stencil::denom)`,
 * where `h` is the grid spacing along that axis. Pre-computing
 * `inv_h2_over_denom = 1/(h^2 * denom)` once per evaluator and applying
 * it after the call is the standard pattern (see `FdGradient<G>`).
 *
 * `Axis` is `0` for x, `1` for y, `2` for z, and chooses which of `sx`,
 * `sy`, `sz` is used as the per-step linear-index stride.
 *
 * **Boundary contract**: caller guarantees that `core[c ± k * stride]`
 * (for `k = 1..half_width`) are valid loads. The point-wise driver
 * (`for_each_interior`) and the brick Laplacian routines achieve this
 * by iterating only over the interior `[hw, n - hw)` slab after a halo
 * exchange with `halo_width >= half_width`.
 *
 * @see fd_stencils.hpp for the stencil tables consumed here
 * @see fd_gradient.hpp for the per-point evaluator that uses these
 *      primitives via the runtime overload
 */

#include <cstddef>
#include <cstdint>

#include <openpfc/kernel/field/fd_stencils.hpp>

namespace pfc::field::fd {

namespace detail {

/// Compile-time selector: returns `sx` if `Axis == 0`, `sy` if `1`, `sz`
/// if `2`. Used by both `apply_d2_along` overloads to lift the per-axis
/// stride decision to compile time.
template <int Axis>
inline std::ptrdiff_t pick_stride(std::ptrdiff_t sx, std::ptrdiff_t sy,
                                  std::ptrdiff_t sz) noexcept {
  static_assert(Axis >= 0 && Axis <= 2, "Axis must be 0 (x), 1 (y), or 2 (z).");
  if constexpr (Axis == 0)
    return sx;
  else if constexpr (Axis == 1)
    return sy;
  else
    return sz;
}

} // namespace detail

/**
 * @brief Apply a compile-time central D2 stencil along `Axis` at point `c`.
 *
 * @tparam Axis    `0` for x, `1` for y, `2` for z.
 * @tparam Stencil A specialization of `EvenCentralD2<Order>` (so
 *                 `half_width`, `denom`, and `coeffs` are
 *                 `static constexpr` and the loop unrolls).
 * @tparam T       Field value type (`double` in normal use).
 *
 * @param core  Base pointer of the local field (row-major, x fastest).
 * @param c     Linear index of the centre cell within `core`.
 * @param sx    Linear-index stride per +1 step in x (typically `1`).
 * @param sy    Stride per +1 step in y (typically `nx`).
 * @param sz    Stride per +1 step in z (typically `nx * ny`).
 *
 * @return Unscaled FD sum
 *         `coeffs[0]*u_c + sum_{k=1..M} coeffs[k]*(u_{c-k*s}+u_{c+k*s})`,
 *         where `s = pick_stride<Axis>(sx, sy, sz)` and `M =
 *         Stencil::half_width`. Multiply by `1 / (h^2 * Stencil::denom)`
 *         to obtain the second derivative along `Axis`.
 *
 * @pre `core[c ± k * s]` for `k = 1..M` are valid loads (caller iterates
 *      only over the interior `[hw, n - hw)` after a halo exchange with
 *      `halo_width >= M`).
 */
template <int Axis, class Stencil, class T>
inline T apply_d2_along(const T *core, std::ptrdiff_t c, std::ptrdiff_t sx,
                        std::ptrdiff_t sy, std::ptrdiff_t sz) noexcept {
  constexpr int M = Stencil::half_width;
  const std::ptrdiff_t s = detail::pick_stride<Axis>(sx, sy, sz);
  T acc = static_cast<T>(Stencil::coeffs[0]) * core[c];
  for (int k = 1; k <= M; ++k) {
    const T ck = static_cast<T>(Stencil::coeffs[k]);
    const std::ptrdiff_t ks = static_cast<std::ptrdiff_t>(k) * s;
    acc += ck * (core[c - ks] + core[c + ks]);
  }
  return acc;
}

/**
 * @brief Apply a runtime central D2 stencil along `Axis` at point `c`.
 *
 * Same arithmetic as the compile-time overload, but reads `half_width`
 * and the coefficient pointer from the runtime view `st` (typically
 * populated via `lookup_even_central_d2(order, &st)`). The per-axis
 * stride decision is still compile-time via `Axis`.
 *
 * @tparam Axis `0` for x, `1` for y, `2` for z.
 * @tparam T    Field value type (`double` in normal use).
 *
 * @param st    Runtime stencil view (must outlive the call; typically
 *              cached on the evaluator at construction time).
 * @param core  Base pointer of the local field (row-major, x fastest).
 * @param c     Linear index of the centre cell within `core`.
 * @param sx    Linear-index stride per +1 step in x.
 * @param sy    Stride per +1 step in y.
 * @param sz    Stride per +1 step in z.
 *
 * @return Same unscaled FD sum as the compile-time overload, with `M =
 *         st.half_width` and weights `st.coeffs[0..M]`.
 */
template <int Axis, class T>
inline T apply_d2_along(const EvenCentralD2View &st, const T *core, std::ptrdiff_t c,
                        std::ptrdiff_t sx, std::ptrdiff_t sy,
                        std::ptrdiff_t sz) noexcept {
  const std::ptrdiff_t s = detail::pick_stride<Axis>(sx, sy, sz);
  const int M = st.half_width;
  const std::int64_t *coeffs = st.coeffs;
  T acc = static_cast<T>(coeffs[0]) * core[c];
  for (int k = 1; k <= M; ++k) {
    const T ck = static_cast<T>(coeffs[k]);
    const std::ptrdiff_t ks = static_cast<std::ptrdiff_t>(k) * s;
    acc += ck * (core[c - ks] + core[c + ks]);
  }
  return acc;
}

} // namespace pfc::field::fd
