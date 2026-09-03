// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file apply_field_modifier.hpp
 * @brief Apply a `FieldModifier` to a canonical `Field` in any memory space.
 *
 * @details
 * Modifiers operate on a host `FieldOutput<double>` over the owned box. For a
 * host-space field this wraps `field.output()` and records the host write; for
 * a device-space field it brackets the call with `Field::with_host_view`, so
 * the framework — not the caller — owns the residency bracket (Audit §4.1 /
 * §13.3). Halo-padded fields are rejected: modifiers see owned cells only.
 */

#include <stdexcept>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>
#include <openpfc/kernel/simulation/simulation_context.hpp>

namespace pfc {

/**
 * @brief Apply @p modifier to @p field at @p time.
 *
 * @param modifier Any `FieldModifier` (IC or BC).
 * @param field    Canonical owning field, host or device space, halo 0.
 * @param time     Simulation time passed to `apply`.
 * @param ctx      Optional simulation context (MPI communicator) for modifiers
 *                 that override the context-taking `apply`.
 * @throws std::invalid_argument if the field has storage halo (modifiers work on
 *         the unpadded owned box).
 */
template <class T, class MemorySpace>
void apply_field_modifier(FieldModifier &modifier, data::Field<T, MemorySpace> &field,
                          double time, const SimulationContext *ctx = nullptr) {
  static_assert(std::is_same_v<T, double>,
                "apply_field_modifier: FieldModifier operates on double fields");
  if (field.storage_halo() != 0) {
    throw std::invalid_argument(
        "apply_field_modifier: field must have storage halo 0 (owned box only)");
  }
  auto call = [&](double *data, std::size_t n) {
    pfc::field::FieldOutput<double> out(data, n);
    if (ctx != nullptr) {
      modifier.apply(*ctx, out, field.domain(), field.box(), time);
    } else {
      modifier.apply(out, field.domain(), field.box(), time);
    }
  };
  if constexpr (data::Field<T, MemorySpace>::is_host_space) {
    call(field.data(), field.size());
    field.note_host_write();
  } else {
    field.with_host_view(call);
  }
}

} // namespace pfc
