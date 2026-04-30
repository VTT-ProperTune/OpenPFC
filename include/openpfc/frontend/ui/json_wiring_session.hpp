// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file json_wiring_session.hpp
 * @brief Bundle MPI/rank context with modifier and results-writer catalogs for JSON
 * wiring
 *
 * @details
 * Drivers and tests pass `JsonWiringContext` together with the catalogs used to
 * resolve JSON `type` / `fields[].writer` strings. There are **no** default
 * catalogs in the constructor: callers name `default_field_modifier_catalog()` /
 * `default_results_writer_catalog()` explicitly at the call site when they intend
 * the process-wide registries (dependency inversion).
 */

#ifndef PFC_UI_JSON_WIRING_SESSION_HPP
#define PFC_UI_JSON_WIRING_SESSION_HPP

#include <mpi.h>

#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/ui/results_writer_catalog.hpp>
#include <openpfc/frontend/ui/simulation_wiring_context.hpp>

namespace pfc::ui {

/**
 * @brief MPI/rank context plus catalogs for JSON wiring
 */
struct JsonWiringSession {
  JsonWiringContext ctx;
  const FieldModifierCatalog &modifier_catalog;
  const ResultsWriterCatalog &writer_catalog;

  JsonWiringSession(JsonWiringContext context,
                    const FieldModifierCatalog &modifier_catalog_in,
                    const ResultsWriterCatalog &writer_catalog_in)
      : ctx(std::move(context)), modifier_catalog(modifier_catalog_in),
        writer_catalog(writer_catalog_in) {}
};

/**
 * @brief Build a session; pass `default_field_modifier_catalog()` /
 *        `default_results_writer_catalog()` when using built-in registrations
 */
[[nodiscard]] inline JsonWiringSession
make_json_wiring_session(MPI_Comm comm, int mpi_rank, bool rank0,
                         const FieldModifierCatalog &modifier_catalog,
                         const ResultsWriterCatalog &writer_catalog) {
  return JsonWiringSession{JsonWiringContext{comm, mpi_rank, rank0},
                           modifier_catalog, writer_catalog};
}

} // namespace pfc::ui

#endif // PFC_UI_JSON_WIRING_SESSION_HPP
