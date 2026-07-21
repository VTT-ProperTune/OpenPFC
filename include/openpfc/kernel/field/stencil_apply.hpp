// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file stencil_apply.hpp
 * @brief **Generic** per-point stencil primitives with runtime/user-supplied
 *        coefficients.
 *
 * @details
 * Sibling to [`fd_apply.hpp`](fd_apply.hpp). Where `fd_apply.hpp` is
 * **PDE-specialised** (it consumes the tabulated `EvenCentralD2<Order>`
 * stencils to compute axis-aligned second derivatives), this header is the
 * **"laboratory" layer**: three primitives that take **arbitrary**
 * coefficient arrays so a user can convolve a row-major 3D field with any
 * stencil they like — first-derivative central FD, Sobel edge-detection,
 * Laplacian-of-Gaussian, anisotropic FD on a non-uniform grid, a learned
 * CNN kernel, etc.
 *
 * Three layers, increasing generality:
 *
 * 1. **`apply_1d_along<Axis>(coeffs, half_width, core, c, sx, sy, sz)`** —
 *    convolve along one axis with a length-`(2*half_width+1)` weight array
 *    indexed `[-hw .. +hw]`. The `EvenCentralD2View`-flavoured FD primitive
 *    is the symmetric-coefficient, integer-weight specialisation of this.
 *
 * 2. **`apply_separable(cx, Hx, cy, Hy, cz, Hz, core, c, sx, sy, sz)`** —
 *    separable tensor-product convolution `cx ⊗ cy ⊗ cz`. Useful when the
 *    3D stencil factors as a product of three 1D stencils (e.g. Sobel
 *    `[-1, 0, +1]_x ⊗ [1, 2, 1]_y ⊗ [1]_z`, Gaussian smoothing, or a mixed
 *    second derivative `∂²/∂x∂y` built from two first-derivative tables).
 *
 * 3. **`apply_dense<Nz, Ny, Nx>(weights, core, c, sx, sy, sz)`** — fully
 *    general dense box stencil with **compile-time** extents (each `Ni`
 *    must be a positive odd integer so the kernel has a well-defined
 *    centre). Use this when the stencil does **not** factor (e.g. some
 *    isotropic Laplacian discretisations on a face-centred-cubic grid,
 *    rotationally-invariant CNN filters, multiscale features).
 *
 * **Boundary contract** (identical to `fd_apply.hpp`): the caller
 * guarantees that every load referenced by the stencil
 * (`core[c + a*sx + b*sy + c*sz]` over the stencil support) lies inside
 * the local backing buffer. The point-wise driver
 * `pfc::sim::for_each_interior` and the manual `for_each_inner /
 * for_each_border` lambdas in `apps/heat3d/src/cpu/heat3d_fd_manual.cpp`
 * achieve this by iterating over `[hw, n - hw)` after a halo exchange of
 * matching width (axis-aligned for `apply_1d_along`; **corner-filled** —
 * see [`full_padded_halo_exchange.hpp`](
 * ../decomposition/full_padded_halo_exchange.hpp) on the host or
 * [`runtime/cuda/full_padded_device_halo.hpp`](
 * ../runtime/cuda/full_padded_device_halo.hpp) on device — for any
 * `apply_separable` or `apply_dense` call whose support extends in more
 * than one axis at once). Face-only `PaddedHaloExchanger` is not
 * sufficient for multi-axis support.
 *
 * **Coefficient layout**: every weight array is **full**, including
 * zero/negative offsets, so the primitives accept asymmetric stencils
 * (Sobel, first-derivative central FD, Roberts cross). The FD primitives
 * in `fd_apply.hpp` exploit symmetry to halve their loop trip count;
 * the primitives here trade that micro-optimisation for genericity.
 *
 * **Type**: every primitive is templated on the field value type `T`
 * (defaults to `double` in normal use). The coefficients are taken with
 * the same `T`. If you need mixed precision (e.g. `float` field with
 * `double` weights), instantiate twice or wrap with explicit casts.
 *
 * @see fd_apply.hpp for the symmetric, integer-weight FD specialisations
 *      (`apply_d2_along`, `apply_tensor_d`).
 * @see fd_stencils.hpp for the tabulated `EvenCentralD2<Order>` weights
 *      consumed by `fd_apply.hpp`.
 * @see openpfc/kernel/simulation/for_each_interior.hpp for the canonical
 *      interior driver loop these primitives plug into.
 * @see runtime/cuda/full_padded_device_halo.hpp for the device corner-filled
 *      halo policy required by `apply_separable` / `apply_dense` cases that
 *      span more than one axis.
 * @see full_padded_halo_exchange.hpp for the host twin
 *      (`pfc::communication::FullPaddedHaloExchanger`).
 */

#include <cstddef>

namespace pfc::field::stencil {

namespace detail {

/// Compile-time stride selector — same trick as
/// `pfc::field::fd::detail::pick_stride`. Duplicated here so this header
/// has no dependency on the FD-specialised stack.
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
 * @brief Convolve a 1D stencil with arbitrary weights along `Axis`.
 *
 * Computes
 * \f[
 *   \mathrm{out} = \sum_{k=-H}^{+H} \mathrm{coeffs}[k+H] \;
 *                  \mathrm{core}[c + k\,s]
 * \f]
 * where `H = half_width` and `s = pick_stride<Axis>(sx, sy, sz)`. The
 * weight array is **full** (length `2*H+1`, indexed `[-H..+H]`); pass
 * `coeffs[H] = 0` for stencils whose centre weight is zero (e.g. central
 * first-derivative).
 *
 * Compared to `pfc::field::fd::apply_d2_along<Axis>` (which assumes
 * symmetric integer coefficients and unscaled output), this primitive:
 *  - accepts **asymmetric** floating-point weights — usable for first
 *    derivatives, Sobel-style edge detection along one axis, or any
 *    custom 1D filter;
 *  - returns the **scaled** result directly (the user has already baked
 *    `1/(denom*h^N)` into `coeffs`).
 *
 * @tparam Axis  `0` for x, `1` for y, `2` for z.
 * @tparam T     Field value type (`double` in normal use).
 *
 * @param coeffs       Pointer to the length-`(2*half_width+1)` weight array,
 *                     indexed `[-half_width .. +half_width]` after the
 *                     `+half_width` shift built into the loop.
 * @param half_width   Stencil radius `H >= 0`; `half_width == 0` returns
 *                     `coeffs[0] * core[c]`.
 * @param core         Base pointer of the local field (row-major, x fastest).
 * @param c            Linear index of the centre cell within `core`.
 * @param sx,sy,sz     Linear-index strides per +1 step in x, y, z.
 *
 * @pre `core[c + k*s]` for `k = -half_width..+half_width` are all valid
 *      loads; the application is responsible for enforcing this (typically
 *      by walking only `[hw, n-hw)` after a halo exchange with the
 *      matching width).
 */
template <int Axis, class T>
inline T apply_1d_along(const T *coeffs, int half_width, const T *core,
                        std::ptrdiff_t c, std::ptrdiff_t sx, std::ptrdiff_t sy,
                        std::ptrdiff_t sz) noexcept {
  const std::ptrdiff_t s = detail::pick_stride<Axis>(sx, sy, sz);
  T acc = T{};
  for (int k = -half_width; k <= half_width; ++k) {
    acc += coeffs[k + half_width] * core[c + static_cast<std::ptrdiff_t>(k) * s];
  }
  return acc;
}

/**
 * @brief Separable tensor-product convolution
 *        \f$ (\mathrm{cx}\otimes \mathrm{cy}\otimes \mathrm{cz}) \star u \f$.
 *
 * Computes
 * \f[
 *   \mathrm{out} = \sum_{a=-H_x}^{H_x}\sum_{b=-H_y}^{H_y}\sum_{c=-H_z}^{H_z}
 *     \mathrm{cx}[a+H_x]\,\mathrm{cy}[b+H_y]\,\mathrm{cz}[c+H_z]\;
 *     \mathrm{core}[c_0 + a\,s_x + b\,s_y + c\,s_z],
 * \f]
 * with `H_i = Hi` (caller-supplied half-widths) and `c_0 = c`.
 *
 * Pure-axis cases collapse cleanly: pass `Hi = 0` and a single weight
 * `ci[0]` (typically `1.0`) to "skip" the stencil along that axis. For
 * example, `apply_separable(sobel_x, 1, smooth, 1, identity, 0, ...)`
 * computes the classical 3×3 Sobel-x edge in 2D embedded in 3D.
 *
 * **Halo requirement**: the union of stencil supports along non-trivial
 * axes generally extends into corner halos. Use a corner-filled exchanger
 * (`pfc::communication::FullPaddedHaloExchanger` on the host, or
 * `pfc::cuda::FullPaddedDeviceHalo` on the GPU) before iterating;
 * `pfc::PaddedHaloExchanger` (axis-aligned only) is **not** sufficient
 * when more than one of `Hi` is non-zero.
 *
 * @tparam T   Field value type.
 *
 * @param cx,cy,cz       Per-axis weight arrays (lengths `2*Hi+1`,
 *                       indexed `[-Hi..+Hi]`).
 * @param Hx,Hy,Hz       Per-axis half-widths (`>= 0`).
 * @param core           Base pointer of the local field (row-major,
 *                       x fastest).
 * @param c              Linear index of the centre cell within `core`.
 * @param sx,sy,sz       Linear-index strides per +1 step in x, y, z.
 */
template <class T>
inline T apply_separable(const T *cx, int Hx, const T *cy, int Hy, const T *cz,
                         int Hz, const T *core, std::ptrdiff_t c, std::ptrdiff_t sx,
                         std::ptrdiff_t sy, std::ptrdiff_t sz) noexcept {
  T acc = T{};
  for (int dz = -Hz; dz <= Hz; ++dz) {
    const T wz = cz[dz + Hz];
    const std::ptrdiff_t offz = static_cast<std::ptrdiff_t>(dz) * sz;
    for (int dy = -Hy; dy <= Hy; ++dy) {
      const T wy = cy[dy + Hy];
      const std::ptrdiff_t offy = static_cast<std::ptrdiff_t>(dy) * sy;
      const T wyz = wy * wz;
      for (int dx = -Hx; dx <= Hx; ++dx) {
        const T wx = cx[dx + Hx];
        const std::ptrdiff_t offx = static_cast<std::ptrdiff_t>(dx) * sx;
        acc += (wx * wyz) * core[c + offx + offy + offz];
      }
    }
  }
  return acc;
}

/**
 * @brief Dense 3D box stencil with compile-time extents.
 *
 * Computes
 * \f[
 *   \mathrm{out} = \sum_{kz=0}^{N_z-1}\sum_{ky=0}^{N_y-1}\sum_{kx=0}^{N_x-1}
 *     \mathrm{weights}[kz][ky][kx]\;
 *     \mathrm{core}[c + (kx-H_x)\,s_x + (ky-H_y)\,s_y + (kz-H_z)\,s_z]
 * \f]
 * where `Hi = (Ni-1)/2` is the per-axis half-width. The stencil extents
 * are template parameters, so the triple loop fully unrolls and the
 * compiler treats every weight as an immediate.
 *
 * Use this when the stencil does **not** separate into a tensor product
 * — for example, a rotationally-invariant 3×3×3 isotropic Laplacian, a
 * multiscale corner-emphasising filter, or a learned CNN kernel. For
 * separable stencils prefer `apply_separable` (cheaper: `Nx*Ny + Nx*Nz +
 * Ny*Nz` vs `Nx*Ny*Nz` flops).
 *
 * @tparam Nz,Ny,Nx  Compile-time stencil extents along z, y, x. Each
 *                   must be a **positive odd integer** so the kernel
 *                   has a well-defined centre.
 * @tparam T         Field value type.
 *
 * @param weights        Reference to a contiguous `[Nz][Ny][Nx]` C-array
 *                       of weights. Indexing is `[kz][ky][kx]` with
 *                       offset `(kx-Hx, ky-Hy, kz-Hz)` from `c`.
 * @param core           Base pointer of the local field (row-major,
 *                       x fastest).
 * @param c              Linear index of the centre cell within `core`.
 * @param sx,sy,sz       Linear-index strides per +1 step in x, y, z.
 *
 * Example — Sobel-x on a 3×3×3 kernel (z averaging trivial):
 * @code
 * constexpr double sobel_x[3][3][3] = {
 *   {{ -1, 0, +1 }, { -2, 0, +2 }, { -1, 0, +1 }},
 *   {{ -2, 0, +2 }, { -4, 0, +4 }, { -2, 0, +2 }},
 *   {{ -1, 0, +1 }, { -2, 0, +2 }, { -1, 0, +1 }},
 * };
 * double gx = pfc::field::stencil::apply_dense(sobel_x, u, c, sx, sy, sz);
 * @endcode
 */
template <int Nz, int Ny, int Nx, class T>
inline T apply_dense(const T (&weights)[Nz][Ny][Nx], const T *core, std::ptrdiff_t c,
                     std::ptrdiff_t sx, std::ptrdiff_t sy,
                     std::ptrdiff_t sz) noexcept {
  static_assert(Nx > 0 && (Nx % 2) == 1,
                "apply_dense: Nx must be a positive odd integer.");
  static_assert(Ny > 0 && (Ny % 2) == 1,
                "apply_dense: Ny must be a positive odd integer.");
  static_assert(Nz > 0 && (Nz % 2) == 1,
                "apply_dense: Nz must be a positive odd integer.");
  constexpr int Hx = (Nx - 1) / 2;
  constexpr int Hy = (Ny - 1) / 2;
  constexpr int Hz = (Nz - 1) / 2;

  T acc = T{};
  for (int kz = 0; kz < Nz; ++kz) {
    const std::ptrdiff_t offz = static_cast<std::ptrdiff_t>(kz - Hz) * sz;
    for (int ky = 0; ky < Ny; ++ky) {
      const std::ptrdiff_t offy = static_cast<std::ptrdiff_t>(ky - Hy) * sy;
      for (int kx = 0; kx < Nx; ++kx) {
        const std::ptrdiff_t offx = static_cast<std::ptrdiff_t>(kx - Hx) * sx;
        acc += weights[kz][ky][kx] * core[c + offx + offy + offz];
      }
    }
  }
  return acc;
}

} // namespace pfc::field::stencil
