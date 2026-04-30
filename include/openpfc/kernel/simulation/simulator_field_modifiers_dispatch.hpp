// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulator_field_modifiers_dispatch.hpp
 * @brief Apply a list of `FieldModifier`s with a shared `SimulationContext`
 *
 * @details
 * Shared by `Simulator::apply_initial_conditions` and
 * `Simulator::apply_boundary_conditions` so the apply-loop lives next to
 * `simulator_results_dispatch.hpp` and stays a single place to extend (e.g.
 * logging or ordering policy).
 *
 * @note This is the production entry point for modifier application: it always
 *       calls `FieldModifier::apply(SimulationContext,...)` (not `apply(Model&,...)`
 *       alone). Keep overrides consistent with the contract documented on
 *       `FieldModifier`.
 */

#ifndef PFC_KERNEL_SIMULATION_SIMULATOR_FIELD_MODIFIERS_DISPATCH_HPP
#define PFC_KERNEL_SIMULATION_SIMULATOR_FIELD_MODIFIERS_DISPATCH_HPP

#include <memory>
#include <vector>

#include <openpfc/kernel/simulation/field_modifier.hpp>
#include <openpfc/kernel/simulation/model.hpp>
#include <openpfc/kernel/simulation/simulation_context.hpp>

namespace pfc {

/**
 * @brief Apply each modifier in registration order at simulation time @p t
 *
 * @param sim_ctx Context passed to `FieldModifier::apply` (MPI comm for I/O, etc.)
 * @param model Target model
 * @param t Current simulation time passed to modifiers
 * @param modifiers Owned modifier list (typically IC or BC bucket from `Simulator`)
 */
inline void apply_field_modifier_list(
    const SimulationContext &sim_ctx, Model &model, double t,
    const std::vector<std::unique_ptr<FieldModifier>> &modifiers) {
  for (const auto &mod : modifiers) {
    mod->apply(sim_ctx, model, t);
  }
}

} // namespace pfc

#endif // PFC_KERNEL_SIMULATION_SIMULATOR_FIELD_MODIFIERS_DISPATCH_HPP
