// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulation_wiring.hpp
 * @brief Connect JSON settings to Simulator and Time (writers, ICs, BCs, options)
 *
 * @details
 * Shared helpers used by `App::main()` (and available for other drivers) to
 * register binary result writers, field modifiers, and optional `simulator`
 * subsection keys (`result_counter`, `increment`).
 *
 * Implementation is split across `simulation_wiring_*.hpp` for readability;
 * including this header pulls in all public APIs.
 *
 * Drivers that do not use `SpectralSimulationSession` can call
 * `add_result_writers_from_json` / `add_initial_conditions_from_json` /
 * `add_boundary_conditions_from_json` and `apply_simulator_section_from_json`
 * individually on an existing `Simulator` and `Time`.
 *
 * Initial-condition and boundary-condition JSON share the same `target`
 * parsing (`configure_field_modifier_targets_from_json`) and the same array
 * wiring loop (`wire_field_modifiers_from_json_array` in
 * `simulation_wiring_conditions.hpp`).
 */

#ifndef PFC_UI_SIMULATION_WIRING_HPP
#define PFC_UI_SIMULATION_WIRING_HPP

#include <openpfc/frontend/ui/simulation_wiring_conditions.hpp>
#include <openpfc/frontend/ui/simulation_wiring_simulator_section.hpp>
#include <openpfc/frontend/ui/simulation_wiring_writers.hpp>

namespace pfc::ui {

/**
 * @brief Writers, ICs, BCs, then optional `simulator` JSON subsection
 *
 * @param modifier_catalog Modifier factories for JSON `type` strings (inject a
 *        test catalog or extend defaults).
 */
inline void
wire_simulator_and_runtime_from_json(Simulator &sim, Time &time,
                                     const nlohmann::json &settings, MPI_Comm comm,
                                     int mpi_rank, bool rank0,
                                     const FieldModifierCatalog &modifier_catalog =
                                         default_field_modifier_catalog()) {
  add_result_writers_from_json(sim, settings, comm, mpi_rank, rank0);
  add_initial_conditions_from_json(sim, settings, comm, mpi_rank, rank0,
                                   modifier_catalog);
  add_boundary_conditions_from_json(sim, settings, comm, mpi_rank, rank0,
                                    modifier_catalog);
  apply_simulator_section_from_json(sim, time, settings);
}

} // namespace pfc::ui

#endif // PFC_UI_SIMULATION_WIRING_HPP
