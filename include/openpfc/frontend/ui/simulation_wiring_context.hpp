// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulation_wiring_context.hpp
 * @brief MPI / rank context shared by JSON simulation wiring helpers
 *
 * @details
 * Bundles the communicator and rank attributes passed to every
 * `add_*_from_json` and `wire_simulator_and_runtime_from_json` entry point so
 * drivers do not repeat five parameters (and tests can construct one object).
 */

#ifndef PFC_UI_SIMULATION_WIRING_CONTEXT_HPP
#define PFC_UI_SIMULATION_WIRING_CONTEXT_HPP

#include <mpi.h>

namespace pfc::ui {

/**
 * @brief Communicator and rank metadata for JSON → `Simulator` wiring
 */
struct JsonWiringContext {
  MPI_Comm comm = MPI_COMM_WORLD;
  int mpi_rank = 0;
  bool rank0 = true;
};

} // namespace pfc::ui

#endif // PFC_UI_SIMULATION_WIRING_CONTEXT_HPP
