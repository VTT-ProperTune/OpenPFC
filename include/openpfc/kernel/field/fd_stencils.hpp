// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file fd_stencils.hpp
 * @brief Compile-time and runtime central FD stencil tables for even orders.
 *
 * @details
 * Tabulates the maximal-accuracy **symmetric** central second-derivative
 * stencils and the **anti-symmetric** central first-derivative stencils on
 * `Order + 1` points for even orders. The shared coefficient layout is
 * \f[
 *   \partial_x^2 u(x_0) \approx
 *     \frac{1}{D_2\,h^2}\Bigl(c^{(2)}_0\,u_0
 *       + \sum_{k=1}^{M} c^{(2)}_k\,(u_{-k}+u_{+k})\Bigr),
 *   \qquad
 *   \partial_x u(x_0) \approx
 *     \frac{1}{D_1\,h}\sum_{k=1}^{M} c^{(1)}_k\,(u_{+k}-u_{-k})
 * \f]
 * with \f$M = \mathrm{Order}/2\f$. The first-derivative weights satisfy the
 * closed form
 * \f$ c^{(1)}_k = (-1)^{k+1}\,(M!)^2 / (k\,(M-k)!\,(M+k)!) \f$ and match the
 * "central, first derivative" Wikipedia table; the second-derivative
 * weights match the "central, second derivative" entry of the same table.
 *
 * Two flavours are exposed for each derivative order:
 *
 * - **Compile-time**: `EvenCentralD1<Order>` / `EvenCentralD2<Order>`
 *   expose `half_width`, `denom`, and `coeffs` as `static constexpr`
 *   members so they can be consumed as template arguments and let the
 *   per-point apply loops in `apply_d1_along<Axis, Stencil>` /
 *   `apply_d2_along<Axis, Stencil>` fully unroll, with the integer
 *   weights becoming immediates.
 *
 * - **Runtime**: `EvenCentralD1View` / `EvenCentralD2View` wrap the same
 *   data behind a small POD handle. `lookup_even_central_d1(order, view)`
 *   / `lookup_even_central_d2(order, view)` populate it from a runtime
 *   `order`, intended for callers that select spatial order from a CLI
 *   or JSON configuration (`apps/heat3d`).
 *
 * **Coverage**: D2 orders 2..20 are tabulated; D1 orders 2..14 are
 * tabulated. Higher D1 orders may be added later as they are needed; the
 * runtime lookup returns `false` for any unsupported order, and
 * `FdGradient<G>` translates that to a clean `std::invalid_argument` when
 * the model declares a derivative member whose stencil is not available.
 *
 * @see fd_apply.hpp for the per-point application primitives that consume
 *      these tables.
 * @see finite_difference.hpp for the brick Laplacian routines that consume
 *      the D2 tables via the runtime view.
 * @see fd_gradient.hpp for the per-point evaluator that uses both tables.
 */

#include <array>
#include <cstdint>

namespace pfc::field::fd {

namespace detail {

/// `static_assert` helper that only fires on instantiation of the primary
/// template (so explicit specializations for valid orders compile cleanly).
template <int> inline constexpr bool always_false_v = false;

} // namespace detail

/**
 * @brief Runtime view onto a central second-derivative stencil.
 *
 * `coeffs[0]` is the central weight; `coeffs[k]` (`1 <= k <= half_width`) is
 * the symmetric weight at offset `±k`. The unscaled finite-difference sum at
 * a point divided by `denom * h^2` yields the second derivative along the
 * stencil's axis.
 */
struct EvenCentralD2View {
  int half_width;
  std::int64_t denom;
  const std::int64_t *coeffs;
};

/**
 * @brief Compile-time central second-derivative stencil for even `Order`.
 *
 * Use as a template argument to `apply_d2_along<Axis, Stencil>` so the
 * stencil loop fully unrolls and the integer weights become immediates.
 * Instantiating with an odd `Order`, or one outside `[2, 20]`, fails at
 * compile time via a `static_assert`.
 */
template <int Order> struct EvenCentralD2 {
  static_assert(detail::always_false_v<Order>,
                "EvenCentralD2: only even orders 2..20 are tabulated.");
};

template <> struct EvenCentralD2<2> {
  static constexpr int half_width = 1;
  static constexpr std::int64_t denom = 1;
  static constexpr std::array<std::int64_t, 2> coeffs = {-2, 1};
};
template <> struct EvenCentralD2<4> {
  static constexpr int half_width = 2;
  static constexpr std::int64_t denom = 12;
  static constexpr std::array<std::int64_t, 3> coeffs = {-30, 16, -1};
};
template <> struct EvenCentralD2<6> {
  static constexpr int half_width = 3;
  static constexpr std::int64_t denom = 180;
  static constexpr std::array<std::int64_t, 4> coeffs = {-490, 270, -27, 2};
};
template <> struct EvenCentralD2<8> {
  static constexpr int half_width = 4;
  static constexpr std::int64_t denom = 5040;
  static constexpr std::array<std::int64_t, 5> coeffs = {-14350, 8064, -1008, 128,
                                                         -9};
};
template <> struct EvenCentralD2<10> {
  static constexpr int half_width = 5;
  static constexpr std::int64_t denom = 25200;
  static constexpr std::array<std::int64_t, 6> coeffs = {-73766, 42000, -6000,
                                                         1000,   -125,  8};
};
template <> struct EvenCentralD2<12> {
  static constexpr int half_width = 6;
  static constexpr std::int64_t denom = 831600;
  static constexpr std::array<std::int64_t, 7> coeffs = {
      -2480478, 1425600, -222750, 44000, -7425, 864, -50};
};
template <> struct EvenCentralD2<14> {
  static constexpr int half_width = 7;
  static constexpr std::int64_t denom = 75675600;
  static constexpr std::array<std::int64_t, 8> coeffs = {
      -228812298, 132432300, -22072050, 4904900, -1003275, 160524, -17150, 900};
};
template <> struct EvenCentralD2<16> {
  static constexpr int half_width = 8;
  static constexpr std::int64_t denom = 302702400;
  static constexpr std::array<std::int64_t, 9> coeffs = {
      -924708642, 538137600, -94174080, 22830080, -5350800,
      1053696,    -156800,   15360,     -735};
};
template <> struct EvenCentralD2<18> {
  static constexpr int half_width = 9;
  static constexpr std::int64_t denom = 15437822400;
  static constexpr std::array<std::int64_t, 10> coeffs = {
      -47541321542, 27788080320, -5052378240, 1309875840, -340063920,
      77728896,     -14394240,   1982880,     -178605,    7840};
};
template <> struct EvenCentralD2<20> {
  static constexpr int half_width = 10;
  static constexpr std::int64_t denom = 293318625600;
  static constexpr std::array<std::int64_t, 11> coeffs = {
      -909151481810, 533306592000, -99994986000, 27349056000,
      -7691922000,   1969132032,   -427329000,   73872000,
      -9426375,      784000,       -31752};
};

/**
 * @brief Populate `*out` with the runtime view of the order-`order` stencil.
 *
 * Thin runtime dispatcher built on the compile-time `EvenCentralD2<Order>`
 * specializations. Intended for callers that pick `order` at runtime (CLI /
 * JSON); compile-time-typed callers should reach for `EvenCentralD2<Order>`
 * directly so the stencil loop unrolls.
 *
 * @return `true` if `order` is one of `2, 4, …, 20`; `false` otherwise (and
 *         `*out` is left untouched).
 */
inline bool lookup_even_central_d2(int order, EvenCentralD2View *out) noexcept {
  switch (order) {
  case 2:
    *out = {EvenCentralD2<2>::half_width, EvenCentralD2<2>::denom,
            EvenCentralD2<2>::coeffs.data()};
    return true;
  case 4:
    *out = {EvenCentralD2<4>::half_width, EvenCentralD2<4>::denom,
            EvenCentralD2<4>::coeffs.data()};
    return true;
  case 6:
    *out = {EvenCentralD2<6>::half_width, EvenCentralD2<6>::denom,
            EvenCentralD2<6>::coeffs.data()};
    return true;
  case 8:
    *out = {EvenCentralD2<8>::half_width, EvenCentralD2<8>::denom,
            EvenCentralD2<8>::coeffs.data()};
    return true;
  case 10:
    *out = {EvenCentralD2<10>::half_width, EvenCentralD2<10>::denom,
            EvenCentralD2<10>::coeffs.data()};
    return true;
  case 12:
    *out = {EvenCentralD2<12>::half_width, EvenCentralD2<12>::denom,
            EvenCentralD2<12>::coeffs.data()};
    return true;
  case 14:
    *out = {EvenCentralD2<14>::half_width, EvenCentralD2<14>::denom,
            EvenCentralD2<14>::coeffs.data()};
    return true;
  case 16:
    *out = {EvenCentralD2<16>::half_width, EvenCentralD2<16>::denom,
            EvenCentralD2<16>::coeffs.data()};
    return true;
  case 18:
    *out = {EvenCentralD2<18>::half_width, EvenCentralD2<18>::denom,
            EvenCentralD2<18>::coeffs.data()};
    return true;
  case 20:
    *out = {EvenCentralD2<20>::half_width, EvenCentralD2<20>::denom,
            EvenCentralD2<20>::coeffs.data()};
    return true;
  default: return false;
  }
}

/**
 * @brief Runtime view onto a central first-derivative stencil.
 *
 * `coeffs[0]` is **always 0** (kept for layout symmetry with
 * `EvenCentralD2View`); `coeffs[k]` for `1 <= k <= half_width` is the
 * weight at offset `+k`. The unscaled finite-difference sum
 * \f$\sum_{k=1}^{M} c_k\,(u_{+k}-u_{-k})\f$ divided by `denom * h` yields
 * the first derivative along the stencil's axis.
 */
struct EvenCentralD1View {
  int half_width;
  std::int64_t denom;
  const std::int64_t *coeffs;
};

/**
 * @brief Compile-time central first-derivative stencil for even `Order`.
 *
 * Use as a template argument to `apply_d1_along<Axis, Stencil>` so the
 * stencil loop fully unrolls. Instantiating with an odd `Order`, or one
 * outside the tabulated range, fails at compile time via a `static_assert`.
 *
 * Coverage: orders 2, 4, 6, 8, 10, 12, 14. Higher orders may be tabulated
 * later if needed; the closed form is in the file-level Doxygen.
 */
template <int Order> struct EvenCentralD1 {
  static_assert(detail::always_false_v<Order>,
                "EvenCentralD1: only even orders 2, 4, 6, 8, 10, 12, 14 are "
                "tabulated.");
};

template <> struct EvenCentralD1<2> {
  static constexpr int half_width = 1;
  static constexpr std::int64_t denom = 2;
  static constexpr std::array<std::int64_t, 2> coeffs = {0, 1};
};
template <> struct EvenCentralD1<4> {
  static constexpr int half_width = 2;
  static constexpr std::int64_t denom = 12;
  static constexpr std::array<std::int64_t, 3> coeffs = {0, 8, -1};
};
template <> struct EvenCentralD1<6> {
  static constexpr int half_width = 3;
  static constexpr std::int64_t denom = 60;
  static constexpr std::array<std::int64_t, 4> coeffs = {0, 45, -9, 1};
};
template <> struct EvenCentralD1<8> {
  static constexpr int half_width = 4;
  static constexpr std::int64_t denom = 840;
  static constexpr std::array<std::int64_t, 5> coeffs = {0, 672, -168, 32, -3};
};
template <> struct EvenCentralD1<10> {
  static constexpr int half_width = 5;
  static constexpr std::int64_t denom = 2520;
  static constexpr std::array<std::int64_t, 6> coeffs = {0, 2100, -600, 150, -25, 2};
};
template <> struct EvenCentralD1<12> {
  static constexpr int half_width = 6;
  static constexpr std::int64_t denom = 27720;
  static constexpr std::array<std::int64_t, 7> coeffs = {0,    23760, -7425, 2200,
                                                         -495, 72,    -5};
};
template <> struct EvenCentralD1<14> {
  static constexpr int half_width = 7;
  static constexpr std::int64_t denom = 360360;
  static constexpr std::array<std::int64_t, 8> coeffs = {
      0, 315315, -105105, 35035, -9555, 1911, -245, 15};
};

/**
 * @brief Populate `*out` with the runtime view of the order-`order` D1 stencil.
 *
 * Thin runtime dispatcher built on the compile-time `EvenCentralD1<Order>`
 * specializations. Returns `true` for orders `2, 4, 6, 8, 10, 12, 14`,
 * `false` otherwise (and `*out` is left untouched).
 */
inline bool lookup_even_central_d1(int order, EvenCentralD1View *out) noexcept {
  switch (order) {
  case 2:
    *out = {EvenCentralD1<2>::half_width, EvenCentralD1<2>::denom,
            EvenCentralD1<2>::coeffs.data()};
    return true;
  case 4:
    *out = {EvenCentralD1<4>::half_width, EvenCentralD1<4>::denom,
            EvenCentralD1<4>::coeffs.data()};
    return true;
  case 6:
    *out = {EvenCentralD1<6>::half_width, EvenCentralD1<6>::denom,
            EvenCentralD1<6>::coeffs.data()};
    return true;
  case 8:
    *out = {EvenCentralD1<8>::half_width, EvenCentralD1<8>::denom,
            EvenCentralD1<8>::coeffs.data()};
    return true;
  case 10:
    *out = {EvenCentralD1<10>::half_width, EvenCentralD1<10>::denom,
            EvenCentralD1<10>::coeffs.data()};
    return true;
  case 12:
    *out = {EvenCentralD1<12>::half_width, EvenCentralD1<12>::denom,
            EvenCentralD1<12>::coeffs.data()};
    return true;
  case 14:
    *out = {EvenCentralD1<14>::half_width, EvenCentralD1<14>::denom,
            EvenCentralD1<14>::coeffs.data()};
    return true;
  default: return false;
  }
}

namespace detail {

/// Back-compat alias used by the legacy non-templated brick routines in
/// `finite_difference.hpp`. Removed in the cleanup commit once those routines
/// migrate to the new `apply_d2_along` primitive.
using EvenFdStencil1d = ::pfc::field::fd::EvenCentralD2View;

/// Back-compat thin forwarder; same removal note as `EvenFdStencil1d`.
inline bool fd_even_order_lookup(int order, EvenFdStencil1d *out) noexcept {
  return ::pfc::field::fd::lookup_even_central_d2(order, out);
}

} // namespace detail

} // namespace pfc::field::fd
