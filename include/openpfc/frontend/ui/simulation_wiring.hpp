// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulation_wiring.hpp
 * @brief Connect JSON settings to Simulator and Time (writers, ICs, BCs, options)
 *
 * @details
 * Shared helpers used by `App::main()` (and available for other drivers) to
 * register result writers, field modifiers, and optional `simulator`
 * subsection keys (`result_counter`, `increment`).
 *
 * Implementation is split across `simulation_wiring_*.hpp` for readability;
 * including this header pulls in all public APIs.
 *
 * Drivers that do not use `SpectralSimulationSession` can call
 * `add_result_writers_from_json` / `add_initial_conditions_from_json` /
 * `add_boundary_conditions_from_json` and `apply_simulator_section_from_json`
 * individually on an existing `Simulator` and `Time`. Pass a `JsonWiringContext`
 * for communicator and rank metadata (see `simulation_wiring_context.hpp`), or a
 * `JsonWiringSession` to bundle context with a `FieldModifierCatalog`
 * (`json_wiring_session.hpp`).
 *
 * Initial-condition and boundary-condition JSON share the same `target`
 * parsing (`configure_field_modifier_targets_from_json`) and the same array
 * wiring loop (`wire_field_modifiers_from_json_array` in
 * `simulation_wiring_conditions.hpp`).
 */

#ifndef PFC_UI_SIMULATION_WIRING_HPP
#define PFC_UI_SIMULATION_WIRING_HPP

#include <openpfc/frontend/ui/json_wiring_session.hpp>
#include <openpfc/frontend/ui/simulation_wiring_conditions.hpp>
#include <openpfc/frontend/ui/simulation_wiring_context.hpp>
#include <openpfc/frontend/ui/simulation_wiring_simulator_section.hpp>
#include <openpfc/frontend/ui/simulation_wiring_writers.hpp>

namespace pfc::ui {

/**
 * @brief Writers, ICs, BCs, then optional `simulator` JSON subsection
 *
 * @details
 * This is a convenience wrapper around four steps (same order). For **partial**
 * wiring or custom ordering, call the underlying functions directly:
 * 1. `add_result_writers_from_json`
 * 2. `add_initial_conditions_from_json`
 * 3. `add_boundary_conditions_from_json`
 * 4. `apply_simulator_section_from_json`
 *
 * @param modifier_catalog Modifier factories for JSON `type` strings (inject a
 *        test catalog or extend defaults).
 * @param writer_catalog Result writer factories for JSON `fields[].writer` (default
 *        `binary`); inject for tests or custom formats.
 *
 * @note **Dependency inversion:** The default arguments use the process-wide
 *       default catalogs (`default_field_modifier_catalog()`,
 *       `default_results_writer_catalog()`). Prefer passing **explicit** catalogs
 *       in tests and when you need isolated registration state.
 */
inline void wire_simulator_and_runtime_from_json(
    Simulator &sim, Time &time, const nlohmann::json &settings,
    const JsonWiringContext &ctx,
    const FieldModifierCatalog &modifier_catalog = default_field_modifier_catalog(),
    const ResultsWriterCatalog &writer_catalog = default_results_writer_catalog()) {
  add_result_writers_from_json(sim, settings, ctx, writer_catalog);
  add_initial_conditions_from_json(sim, settings, ctx, modifier_catalog);
  add_boundary_conditions_from_json(sim, settings, ctx, modifier_catalog);
  apply_simulator_section_from_json(sim, time, settings);
}

inline void wire_simulator_and_runtime_from_json(
    Simulator &sim, Time &time, const nlohmann::json &settings, MPI_Comm comm,
    int mpi_rank, bool rank0,
    const FieldModifierCatalog &modifier_catalog = default_field_modifier_catalog(),
    const ResultsWriterCatalog &writer_catalog = default_results_writer_catalog()) {
  wire_simulator_and_runtime_from_json(sim, time, settings,
                                       JsonWiringContext{comm, mpi_rank, rank0},
                                       modifier_catalog, writer_catalog);
}

/**
 * @brief Same as `wire_simulator_and_runtime_from_json(sim, time, settings, ctx,
 *        catalog)` with `ctx` and `catalog` taken from `session`
 */
inline void wire_simulator_and_runtime_from_json(Simulator &sim, Time &time,
                                                 const nlohmann::json &settings,
                                                 const JsonWiringSession &session) {
  wire_simulator_and_runtime_from_json(sim, time, settings, session.ctx,
                                       session.modifier_catalog);
}

/**
 * @brief Same as `wire_simulator_and_runtime_from_json` with `session` for context
 *        and modifier catalog, but with an explicit results-writer catalog
 *
 * Use this in tests (or custom drivers) when you inject a `JsonWiringSession` for
 * MPI/modifier factories but still need a non-default `ResultsWriterCatalog`.
 */
inline void wire_simulator_and_runtime_from_json(
    Simulator &sim, Time &time, const nlohmann::json &settings,
    const JsonWiringSession &session, const ResultsWriterCatalog &writer_catalog) {
  wire_simulator_and_runtime_from_json(sim, time, settings, session.ctx,
                                       session.modifier_catalog, writer_catalog);
}

} // namespace pfc::ui

#endif // PFC_UI_SIMULATION_WIRING_HPP
