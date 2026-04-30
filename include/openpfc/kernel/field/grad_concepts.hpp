// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file grad_concepts.hpp
 * @brief C++20 concepts that introspect a model's per-point grads aggregate.
 *
 * @details
 * The point-wise driver loop (`pfc::sim::for_each_interior`) calls
 *
 *     du[{i,j,k}] = model.rhs(t, eval(i,j,k))
 *
 * where `eval(i,j,k)` returns a *model-owned* aggregate that names exactly
 * the partial derivatives the model needs from the catalog
 * `{ value, x, y, z, xx, yy, zz, xy, xz, yz }`.
 *
 * The concepts in this header let a backend evaluator (FD, spectral, GPU)
 * inspect that aggregate at compile time and produce only the members the
 * model actually consumes — no wasted FFTs, no wasted stencil sweeps.
 * Members the backend cannot supply (e.g. `xy` for a face-halo-only FD
 * stencil) trigger a `static_assert` at the call site rather than silently
 * producing zeros.
 *
 * Example:
 * @code
 * struct HeatGrads { double xx{}, yy{}, zz{}; };
 *
 * template <class G>
 * G FdGradient<G>::operator()(int ix, int iy, int iz) const noexcept {
 *   G g{};
 *   if constexpr (pfc::field::has_value<G>) g.value = ...;
 *   if constexpr (pfc::field::has_xx<G>)    g.xx    = ...;
 *   // ...
 *   return g;
 * }
 * @endcode
 *
 * @see grad_point.hpp for the default catalog struct apps can reach for
 *      when they don't want to declare their own.
 * @see openpfc/kernel/simulation/for_each_interior.hpp for the driver loop
 */

namespace pfc::field {

/** True when the per-point grads aggregate `G` has a `.value` member. */
template <class G>
concept has_value = requires(G &g) { g.value; };

/** True when `G` has a first-derivative member along x. */
template <class G>
concept has_x = requires(G &g) { g.x; };

/** True when `G` has a first-derivative member along y. */
template <class G>
concept has_y = requires(G &g) { g.y; };

/** True when `G` has a first-derivative member along z. */
template <class G>
concept has_z = requires(G &g) { g.z; };

/** True when `G` has the second derivative `xx`. */
template <class G>
concept has_xx = requires(G &g) { g.xx; };

/** True when `G` has the second derivative `yy`. */
template <class G>
concept has_yy = requires(G &g) { g.yy; };

/** True when `G` has the second derivative `zz`. */
template <class G>
concept has_zz = requires(G &g) { g.zz; };

/** True when `G` has the mixed derivative `xy`. */
template <class G>
concept has_xy = requires(G &g) { g.xy; };

/** True when `G` has the mixed derivative `xz`. */
template <class G>
concept has_xz = requires(G &g) { g.xz; };

/** True when `G` has the mixed derivative `yz`. */
template <class G>
concept has_yz = requires(G &g) { g.yz; };

} // namespace pfc::field
