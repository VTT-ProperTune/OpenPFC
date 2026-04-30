// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file fd_stencils.hpp
 * @brief Compile-time and runtime central FD stencil tables for even orders.
 *
 * @details
 * Tabulates the maximal-accuracy central second-derivative stencils on
 * `Order + 1` points for even orders `2, 4, …, 20`. The integer coefficients
 * \f$c_k\f$ and shared denominator \f$D\f$ satisfy
 * \f[
 *   \partial_x^2 u(x_0) \approx
 *     \frac{1}{D\,h^2}\Bigl(c_0\,u_0 + \sum_{k=1}^{M} c_k\,(u_{-k}+u_{+k})\Bigr)
 * \f]
 * with \f$M = \mathrm{Order}/2\f$. Numbers match the "central, second
 * derivative" table on Wikipedia ("Finite difference coefficient").
 *
 * Two flavours are exposed:
 *
 * - **Compile-time**: `EvenCentralD2<Order>` exposes `half_width`, `denom`,
 *   and `coeffs` as `static constexpr` members so they can be consumed as
 *   template arguments and let the per-point apply loop in
 *   `apply_d2_along<Axis, Stencil>` fully unroll, with the integer weights
 *   becoming immediates.
 *
 * - **Runtime**: `EvenCentralD2View` wraps the same data behind a small POD
 *   handle. `lookup_even_central_d2(order, view)` populates it from a runtime
 *   `order`, intended for callers that select spatial order from a CLI or
 *   JSON configuration (`apps/heat3d`).
 *
 * @see fd_apply.hpp for the per-point application primitives that consume
 *      these tables (added in a follow-up commit).
 * @see finite_difference.hpp for the brick Laplacian routines that consume
 *      these tables today via the runtime view.
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
