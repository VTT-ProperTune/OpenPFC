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
 * (`[nx, ny, nz]`, x varies fastest). Two layers are exposed:
 *
 * 1. **Per-axis** `apply_d2_along<Axis, Stencil>(core, c, sx, sy, sz)`
 *    (and its runtime-stencil sibling) -- the workhorse for pure-axis
 *    second derivatives. With `Stencil = EvenCentralD2<Order>` the
 *    coefficient loop unrolls; the runtime overload takes a
 *    `EvenCentralD2View` for callers that pick `Order` from a CLI /
 *    JSON configuration (`apps/heat3d`). Both forms return the
 *    **unscaled** finite-difference sum: divide by `h^2 *
 *    Stencil::denom` to obtain the second derivative.
 *
 * 2. **Tensor-product** `apply_tensor_d<Mx, My, Mz, Sx, Sy, Sz>(core, c,
 *    sx, sy, sz)` -- the explicit Cartesian-product form
 *    \f$ \partial_x^{M_x} \partial_y^{M_y} \partial_z^{M_z} u_{i,j,k}
 *      \approx \frac{1}{D_x D_y D_z\,h_x^{M_x} h_y^{M_y} h_z^{M_z}}
 *      \sum_{a,b,c} c^x_a c^y_b c^z_c\, u_{i+a, j+b, k+c} \f$.
 *    Pure-axis cases reduce to `apply_d2_along`; mixed second
 *    derivatives (e.g. `Mx=My=2, Mz=0` for the future `xxyy` member)
 *    fall out as a single triple loop. Today only `Mi in {0, 2}` is
 *    accepted (no first-derivative tables yet); enabling Mi=1 just
 *    adds an `EvenCentralD1<Order>` table family, no other code
 *    changes.
 *
 * `Axis` is `0` for x, `1` for y, `2` for z, and chooses which of `sx`,
 * `sy`, `sz` is used as the per-step linear-index stride.
 *
 * **Boundary contract**: caller guarantees that every load referenced
 * by the stencil (offsets `k * stride` for the per-axis form, the full
 * cuboid `[-Hx, Hx] x [-Hy, Hy] x [-Hz, Hz]` for the tensor form) lies
 * inside the local backing buffer. The point-wise driver
 * (`for_each_interior`) and the brick Laplacian routines achieve this
 * for pure-axis stencils by iterating over `[hw, n - hw)` after a halo
 * exchange with `halo_width >= half_width`. Mixed-second tensor
 * products additionally need corner halos -- not yet shipped by the
 * exchanger, which is why `FdGradient<G>` rejects `xy / xz / yz` at
 * compile time even though the primitive itself is ready.
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

/**
 * @brief Identity (Mi = 0) stencil: contributes one cell at offset 0.
 *
 * Used as the per-axis stencil placeholder in `apply_tensor_d` for axes
 * along which the requested derivative order is zero. The interface
 * (`half_width`, `denom`, `coeffs`) matches `EvenCentralD2<Order>` so
 * the tensor-product loop body is uniform across all combinations.
 */
struct IdentityStencil1d {
  static constexpr int half_width = 0;
  static constexpr std::int64_t denom = 1;
  static constexpr std::array<std::int64_t, 1> coeffs = {1};
};

/**
 * @brief Apply a tensor-product central FD stencil at point `c`.
 *
 * Computes the unscaled Cartesian-product sum
 * \f[
 *   \sum_{a=-H_x}^{H_x}\sum_{b=-H_y}^{H_y}\sum_{c=-H_z}^{H_z}
 *     c^x_{|a|}\,c^y_{|b|}\,c^z_{|c|}\;
 *     \mathrm{core}[c + a\,s_x + b\,s_y + c\,s_z]
 * \f]
 * where `H_i = StencilI::half_width` and `c^i_k = StencilI::coeffs[k]`.
 * The corresponding mixed partial derivative is the result divided by
 * `StencilX::denom * StencilY::denom * StencilZ::denom * h_x^Mx *
 * h_y^My * h_z^Mz`.
 *
 * Pure-axis cases (e.g. `Mx=2, My=Mz=0`) collapse the two `Mi=0` loops
 * to a single `dki=0` iteration with weight `1` and reduce exactly to
 * `apply_d2_along<X-axis, StencilX>(core, c, sx, sy, sz)`. Reach for
 * `apply_d2_along` directly in that case to skip the cuboid-loop
 * boilerplate.
 *
 * @tparam Mx,My,Mz   Per-axis derivative order. Must each be `0` or `2`
 *                    today; first-derivative tables (`Mi = 1`) are not
 *                    yet implemented and trigger a `static_assert`.
 * @tparam StencilX,Y,Z  Per-axis stencil type. Pass `IdentityStencil1d`
 *                    (the default) for axes with `Mi = 0`; pass
 *                    `EvenCentralD2<Order>` for axes with `Mi = 2`.
 * @tparam T          Field value type (`double` in normal use).
 *
 * @param core  Base pointer of the local field (row-major, x fastest).
 * @param c     Linear index of the centre cell within `core`.
 * @param sx,sy,sz  Linear-index strides per +1 step in x, y, z.
 *
 * @pre Every load `core[c + a*sx + b*sy + c*sz]` for
 *      `(a,b,c) in [-Hx,Hx] x [-Hy,Hy] x [-Hz,Hz]` is in-bounds. For
 *      mixed-second cases (>= 2 non-zero `Mi`), this requires corner
 *      halos -- not currently shipped by the halo exchanger, hence
 *      `FdGradient<G>` keeps its `static_assert` against `xy/xz/yz`
 *      members even though this primitive is ready.
 */
template <int Mx, int My, int Mz, class StencilX = IdentityStencil1d,
          class StencilY = IdentityStencil1d, class StencilZ = IdentityStencil1d,
          class T>
inline T apply_tensor_d(const T *core, std::ptrdiff_t c, std::ptrdiff_t sx,
                        std::ptrdiff_t sy, std::ptrdiff_t sz) noexcept {
  static_assert(Mx == 0 || Mx == 2,
                "apply_tensor_d: Mx must be 0 or 2; first-derivative "
                "tables are not yet implemented.");
  static_assert(My == 0 || My == 2,
                "apply_tensor_d: My must be 0 or 2; first-derivative "
                "tables are not yet implemented.");
  static_assert(Mz == 0 || Mz == 2,
                "apply_tensor_d: Mz must be 0 or 2; first-derivative "
                "tables are not yet implemented.");

  constexpr int Hx = StencilX::half_width;
  constexpr int Hy = StencilY::half_width;
  constexpr int Hz = StencilZ::half_width;

  T acc{};
  for (int dkz = -Hz; dkz <= Hz; ++dkz) {
    const std::int64_t cz = StencilZ::coeffs[(dkz < 0) ? -dkz : dkz];
    const std::ptrdiff_t offz = static_cast<std::ptrdiff_t>(dkz) * sz;
    for (int dky = -Hy; dky <= Hy; ++dky) {
      const std::int64_t cy = StencilY::coeffs[(dky < 0) ? -dky : dky];
      const std::ptrdiff_t offy = static_cast<std::ptrdiff_t>(dky) * sy;
      const std::int64_t cyz = cy * cz;
      for (int dkx = -Hx; dkx <= Hx; ++dkx) {
        const std::int64_t cx = StencilX::coeffs[(dkx < 0) ? -dkx : dkx];
        const std::ptrdiff_t offx = static_cast<std::ptrdiff_t>(dkx) * sx;
        acc += static_cast<T>(cx * cyz) * core[c + offx + offy + offz];
      }
    }
  }
  return acc;
}

} // namespace pfc::field::fd
