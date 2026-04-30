// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file grad_point.hpp
 * @brief Local partial derivatives of a scalar field at one cell.
 *
 * @details
 * `GradPoint` is the **interface contract** between a per-point right-hand
 * side (RHS) function and a spatial-discretization evaluator. A model RHS of
 * the canonical shape
 *
 *     du[{i,j,k}] = rhs(t, GradPoint{...})
 *
 * works unchanged with any backend that can fill the requested fields:
 *  - finite-difference evaluator (`pfc::field::FdGradient`)
 *  - spectral evaluator (`pfc::field::SpectralGradient`)
 *  - GPU evaluators (future)
 *
 * Today's `GradPoint` carries only what current OpenPFC apps consume:
 * `u`, `uxx`, `uyy`, `uzz`. Mixed terms (`uxy`, `uxz`, `uyz`) require corner
 * halos on the FD side and are intentionally out of scope here. Extend this
 * struct (and the evaluators) when a new model needs additional partials.
 *
 * @see fd_gradient.hpp for the FD evaluator
 * @see spectral_gradient.hpp for the spectral evaluator
 * @see openpfc/kernel/simulation/for_each_interior.hpp for the driver loop
 */

namespace pfc::field {

/**
 * @brief Local partial derivatives of a scalar field at one interior cell.
 *
 * Default-constructed members are zero so an evaluator that fills only a
 * subset still produces a well-defined value for unfilled members.
 */
struct GradPoint {
  double u{};
  double uxx{};
  double uyy{};
  double uzz{};
};

} // namespace pfc::field
