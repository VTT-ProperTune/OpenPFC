// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulation_wiring.hpp
 * @brief Connect JSON settings to Time, FieldModifiers, and ResultsWriters
 *
 * @details
 * Shared helpers used by 0.2 sessions and drivers to parse result writers,
 * field modifiers, and optional `simulator` subsection keys (`increment`,
 * `integrator.method`) plus JSON `restart_from` exclusivity.
 *
 * Implementation is split across `simulation_wiring_*.hpp` for readability;
 * including this header pulls in all public APIs.
 *
 * Drivers call `parse_result_writers_from_json` /
 * `parse_initial_conditions_from_json` / `parse_boundary_conditions_from_json`
 * and `apply_simulator_section_from_json` on `Time`. Pass a `JsonWiringContext`
 * for communicator and rank metadata (see `simulation_wiring_context.hpp`), or a
 * `JsonWiringSession` to bundle context with modifier and results-writer catalogs
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
 * @brief Parsed JSON writers, ICs, and BCs (Time overlays applied separately)
 */
struct JsonRuntimeWiring {
  std::vector<std::unique_ptr<FieldModifier>> initial_conditions;
  std::vector<std::unique_ptr<FieldModifier>> boundary_conditions;
  std::vector<NamedResultsWriter> writers;
};

/**
 * @brief Writers, ICs, BCs, then optional `simulator` JSON subsection
 *
 * @details
 * This is a convenience wrapper around four steps (same order). For **partial**
 * wiring or custom ordering, call the underlying functions directly:
 * 1. `parse_result_writers_from_json`
 * 2. `parse_initial_conditions_from_json`
 * 3. `parse_boundary_conditions_from_json`
 * 4. `apply_simulator_section_from_json`
 *
 * @param modifier_catalog Modifier factories for JSON `type` strings.
 * @param writer_catalog Result writer factories for JSON `fields[].writer` (e.g.
 *        `default_results_writer_catalog()` for built-in `binary`).
 *
 * @note **Dependency inversion:** Both catalogs are **required** parameters—there
 *       are no hidden defaults. Use `default_field_modifier_catalog()` /
 *       `default_results_writer_catalog()` at the call site when you intend the
 *       process-wide registries.
 */
inline JsonRuntimeWiring parse_runtime_from_json(
    Time &time, const nlohmann::json &settings, const JsonWiringContext &ctx,
    const FieldModifierCatalog &modifier_catalog,
    const ResultsWriterCatalog &writer_catalog) {
  JsonRuntimeWiring wiring;
  wiring.writers = parse_result_writers_from_json(settings, ctx, writer_catalog);
  wiring.initial_conditions =
      parse_initial_conditions_from_json(settings, ctx, modifier_catalog);
  wiring.boundary_conditions =
      parse_boundary_conditions_from_json(settings, ctx, modifier_catalog);
  apply_simulator_section_from_json(time, settings);
  return wiring;
}

/**
 * @brief Same as `parse_runtime_from_json(time, settings, ctx, catalogs)` with
 *        `ctx` and catalogs taken from `session`
 */
inline JsonRuntimeWiring parse_runtime_from_json(Time &time,
                                                 const nlohmann::json &settings,
                                                 const JsonWiringSession &session) {
  return parse_runtime_from_json(time, settings, session.ctx,
                                 session.modifier_catalog, session.writer_catalog);
}

} // namespace pfc::ui

#endif // PFC_UI_SIMULATION_WIRING_HPP
