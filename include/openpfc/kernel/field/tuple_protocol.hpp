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
 * @see openpfc/kernel/simulation/for_each_interior.hpp for the consumer
 * @see openpfc/kernel/field/composite_gradient.hpp for multi-field eval
 */

#include <tuple>
#include <type_traits>

namespace pfc::field::detail {

/** True iff `T` opts in to the protocol via a `t.as_tuple()` member. */
template <class T>
concept has_as_tuple = requires(T &t) { t.as_tuple(); };

template <class T> struct is_std_tuple : std::false_type {};

template <class... Ts> struct is_std_tuple<std::tuple<Ts...>> : std::true_type {};

/** True iff `T` is a `std::tuple` specialization. */
template <class T>
concept is_tuple = is_std_tuple<std::remove_cvref_t<T>>::value;

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
