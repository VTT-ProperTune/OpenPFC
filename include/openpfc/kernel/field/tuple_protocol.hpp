// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file tuple_protocol.hpp
 * @brief Tiny in-tree alternative to `boost::pfr` for multi-field bundles.
 *
 * @details
 * Multi-field point-wise models look like
 *
 *     auto inc = model.rhs(t, composite_grads);
 *
 * where `composite_grads` bundles per-field grads structs (e.g. `g.u`,
 * `g.v`, `g.w`) and `inc` bundles per-field increments (`du`, `dv`,
 * `dw`). The driver loop must scatter `inc` into the right `du[k]`
 * slot in the right rank-local buffer for each field.
 *
 * Boost.PFR would solve this with magic structured-binding reflection.
 * To avoid an external dependency we adopt a one-line opt-in convention:
 *
 *  - A user struct may expose `auto as_tuple() &` (and a `const &` overload)
 *    that returns a `std::tuple` of references to its members in the order
 *    the framework should iterate them.
 *  - `std::tuple` itself is also accepted and used as-is.
 *  - A bare scalar (`double`, `int`, ...) is treated as a single-field
 *    bundle and wrapped in a one-tuple.
 *
 * Detection uses C++17 SFINAE traits (`struct has_as_tuple` /
 * `struct is_tuple`) rather than C++20 `concept` / `requires`, so this
 * header remains includable from `.cu` translation units when nvcc falls
 * back to a C++17 host dialect. Boolean variable templates with the same
 * names preserve the former concept call syntax (`has_as_tuple<T>` in
 * `static_assert` / `if constexpr`) for existing host call sites.
 *
 * Example:
 * @code
 * struct WaveLocal {
 *   UGrads u;
 *   VGrads v;
 *   auto as_tuple()       { return std::tie(u, v); }
 *   auto as_tuple() const { return std::tie(u, v); }
 * };
 * @endcode
 *
 * @note `to_tuple` is **host-oriented**. Device multi-field scatter uses
 *       `DevicePtrPackN` / `scatter_device` in
 *       `runtime/cuda/for_each_interior_device.hpp` and must not call
 *       `to_tuple`, `std::get`, `std::apply`, or `std::forward_as_tuple`
 *       from `__device__` code.
 *
 * @see openpfc/kernel/simulation/for_each_interior.hpp for the consumer
 * @see openpfc/kernel/field/composite_gradient.hpp for multi-field eval
 */

#include <tuple>
#include <type_traits>
#include <utility>

namespace pfc::field::detail {

namespace sfinae {

/** True iff `T` opts in to the protocol via a `t.as_tuple()` member. */
template <class T, class = void>
struct has_as_tuple : std::false_type {};

template <class T>
struct has_as_tuple<T, std::void_t<decltype(std::declval<T &>().as_tuple())>>
    : std::true_type {};

template <class T> struct is_std_tuple : std::false_type {};

template <class... Ts> struct is_std_tuple<std::tuple<Ts...>> : std::true_type {};

/** True iff `T` is a `std::tuple` specialization (cv/ref stripped). */
template <class T>
struct is_tuple : is_std_tuple<std::remove_cv_t<std::remove_reference_t<T>>> {};

} // namespace sfinae

/**
 * Boolean predicates (same names as the former C++20 concepts).
 * Prefer these in `if constexpr` / `static_assert`; the SFINAE structs
 * live in `sfinae::` when `::value` is needed explicitly.
 */
template <class T>
inline constexpr bool has_as_tuple = sfinae::has_as_tuple<T>::value;

template <class T>
inline constexpr bool is_tuple = sfinae::is_tuple<T>::value;

/**
 * @brief Normalize `t` into a tuple-like view for fan-out.
 *
 * Returns `t.as_tuple()` if `T` opts in, `t` itself if it is already a
 * `std::tuple`, otherwise `std::forward_as_tuple(t)` (one-element view).
 */
template <class T> constexpr decltype(auto) to_tuple(T &t) {
  if constexpr (has_as_tuple<T>) {
    return t.as_tuple();
  } else if constexpr (is_tuple<T>) {
    return (t);
  } else {
    return std::forward_as_tuple(t);
  }
}

template <class T> constexpr decltype(auto) to_tuple(const T &t) {
  if constexpr (has_as_tuple<T>) {
    return t.as_tuple();
  } else if constexpr (is_tuple<T>) {
    return (t);
  } else {
    return std::forward_as_tuple(t);
  }
}

} // namespace pfc::field::detail
