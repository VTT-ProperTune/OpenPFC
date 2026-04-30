// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file heat_model.hpp
 * @brief Heat3D physical model and point-wise RHS.
 *
 * Separates concerns the way the design discussion (and the Jugru-style
 * canonical form `f(model, cache, du, u, Ω, t)`) calls for:
 *
 *  - `HeatModel` is **data only** — physical parameters of the PDE.
 *  - `HeatRhs` is a **callable struct** that holds a `HeatModel` and evaluates
 *    the right-hand side at a single interior point given a `GradPoint`.
 *
 * The same `HeatRhs` works under any backend that can produce a `GradPoint`
 * (today: `FdGradient`; tomorrow: a spectral evaluator).
 */

#include <heat3d/discretization.hpp>

namespace heat3d {

/** Physical parameters of the 3D heat equation \f$\partial_t u = D\nabla^2 u\f$. */
struct HeatModel {
  double D{1.0};
};

/**
 * @brief Point-wise RHS for \f$\partial_t u = D\nabla^2 u\f$.
 *
 * `inline`/`noexcept` so the surrounding `for_each_interior` driver can fuse
 * stencil + RHS into one tight inner loop.
 */
struct HeatRhs {
  HeatModel model{};

  inline double operator()(double /*t*/, const GradPoint &g) const noexcept {
    return model.D * (g.uxx + g.uyy + g.uzz);
  }
};

} // namespace heat3d
