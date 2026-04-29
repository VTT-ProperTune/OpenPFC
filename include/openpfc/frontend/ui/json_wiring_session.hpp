// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file json_wiring_session.hpp
 * @brief Bundle MPI/rank context with a `FieldModifierCatalog` for JSON wiring
 *
 * @details
 * Drivers and tests often pass the same `JsonWiringContext` and modifier catalog
 * together into `wire_simulator_and_runtime_from_json`. This struct keeps that
 * pair in one object (optional catalog injection for unit tests).
 */

#ifndef PFC_UI_JSON_WIRING_SESSION_HPP
#define PFC_UI_JSON_WIRING_SESSION_HPP

#include <mpi.h>

#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/ui/simulation_wiring_context.hpp>

namespace pfc::ui {

/**
 * @brief MPI/rank context plus field-modifier factories for JSON `type` strings
 */
struct JsonWiringSession {
  JsonWiringContext ctx;
  const FieldModifierCatalog &modifier_catalog;

  JsonWiringSession(JsonWiringContext context, const FieldModifierCatalog &catalog =
                                                   default_field_modifier_catalog())
      : ctx(std::move(context)), modifier_catalog(catalog) {}
};

/**
 * @brief Convenience factory (default catalog = built-in + registered types)
 */
[[nodiscard]] inline JsonWiringSession
make_json_wiring_session(MPI_Comm comm, int mpi_rank, bool rank0,
                         const FieldModifierCatalog &modifier_catalog =
                             default_field_modifier_catalog()) {
  return JsonWiringSession{JsonWiringContext{comm, mpi_rank, rank0},
                           modifier_catalog};
}

} // namespace pfc::ui

#endif // PFC_UI_JSON_WIRING_SESSION_HPP
