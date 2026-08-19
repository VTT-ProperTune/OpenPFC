// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file legacy_model_physics.hpp
 * @brief Adapter A1: wrap a Gen-1 `Model` as `SteppablePhysics`.
 *
 * @details
 * Temporary (M7 → M12). New code must not depend on this adapter.
 * `step(t)` forwards to `Model::step(t)` so `Simulator::step_with_physics`
 * (A2) can drive a legacy model through the concept surface.
 *
 * `declare_fields` is a no-op: Gen-1 fields live on `ModelFieldRegistry`,
 * not `SimulationState`.
 *
 * @see docs/development/0.2_migration_map.md
 */

#include <openpfc/kernel/simulation/model.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>

namespace pfc::compat {

/**
 * @brief Non-owning wrapper of a Gen-1 `Model` as concept physics.
 *
 * The referenced `Model` must outlive this object.
 */
class LegacyModelPhysics {
public:
  explicit LegacyModelPhysics(Model &model) noexcept : m_model(&model) {}

  void step(double t) { m_model->step(t); }

  void declare_fields(SimulationState &) const {}

  [[nodiscard]] Model &model() noexcept { return *m_model; }
  [[nodiscard]] const Model &model() const noexcept { return *m_model; }

private:
  Model *m_model;
};

static_assert(pfc::sim::SteppablePhysics<LegacyModelPhysics>);
static_assert(pfc::sim::DeclaresFields<LegacyModelPhysics>);

} // namespace pfc::compat
