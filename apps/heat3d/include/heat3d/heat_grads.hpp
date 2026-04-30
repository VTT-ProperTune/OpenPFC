// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file heat_grads.hpp
 * @brief Per-point grads aggregate consumed by `heat3d::HeatModel::rhs`.
 *
 * @details
 * The heat equation \f$\partial_t u = D\nabla^2 u\f$ needs only the three
 * unmixed second derivatives of `u`, so we declare a minimal aggregate
 * naming exactly those slots from the catalog
 * `{ value, x, y, z, xx, yy, zz, xy, xz, yz }` recognized by the
 * OpenPFC kernel.
 *
 * Because the per-point evaluators (`pfc::field::FdGradient<G>`,
 * `pfc::field::SpectralGradient<G>`) are templated on `G` and use the
 * `pfc::field::has_*` concepts to fill only the members `G` declares,
 * an evaluator instantiated with `HeatGrads` computes only `xx`, `yy`,
 * and `zz` — no wasted stencil sweeps, no wasted FFTs.
 *
 * This struct is intentionally trivial and free of any `<openpfc/...>`
 * dependency so that `heat_model.hpp` stays a pure-physics header.
 *
 * @see openpfc/kernel/field/grad_concepts.hpp for the per-member concepts
 * @see openpfc/kernel/field/grad_point.hpp for the convenience default
 *      catalog struct apps can use instead when minimizing kernel work
 *      isn't critical
 */

namespace heat3d {

/**
 * @brief Minimal per-point grads aggregate for the heat equation.
 *
 * Default-initialized to zero so any evaluator that fails to populate a
 * slot (today: none — both FD and spectral fill every member declared
 * here) still yields a well-defined RHS evaluation.
 */
struct HeatGrads {
  double xx{};
  double yy{};
  double zz{};
};

} // namespace heat3d
