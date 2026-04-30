// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file grad_point.hpp
 * @brief Default catalog struct for the per-point grads bundle.
 *
 * @details
 * `GradPoint` is the **default convenience aggregate** an OpenPFC application
 * can pass to a per-point evaluator when it does not want to declare its own
 * model-specific grads struct. It enumerates the full catalog of partial
 * derivatives recognized by the kernel:
 *
 *   `value, x, y, z, xx, yy, zz, xy, xz, yz`
 *
 * The introspection layer (see `grad_concepts.hpp`) detects each member
 * individually, so backends populate only the slots a model actually reads
 * — using `GradPoint` does **not** force every backend to compute every
 * derivative on every step. Apps that want the cheapest possible bundle
 * should still declare a minimal model-owned struct (e.g.
 * `struct HeatGrads { double xx{}, yy{}, zz{}; };`) — the kernel works
 * identically on either type.
 *
 * Default-initialized members are zero so an evaluator that fills only a
 * subset still produces a well-defined value for unfilled members.
 *
 * @see grad_concepts.hpp for the per-member detection concepts
 * @see fd_gradient.hpp for the FD evaluator
 * @see spectral_gradient.hpp for the spectral evaluator
 * @see openpfc/kernel/simulation/for_each_interior.hpp for the driver loop
 */

namespace pfc::field {

/**
 * @brief Default per-point grads aggregate (full catalog).
 *
 * Using this struct is purely a convenience — backends still introspect
 * each member individually via `pfc::field::has_*` concepts and only
 * populate the slots that exist. A model-defined struct that names only
 * the members it needs is preferred when minimizing kernel work matters.
 */
struct GradPoint {
  double value{};
  double x{};
  double y{};
  double z{};
  double xx{};
  double yy{};
  double zz{};
  double xy{};
  double xz{};
  double yz{};
};

} // namespace pfc::field
