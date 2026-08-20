// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file from_json_simulation_session.hpp
 * @brief JSON `method`/`backend`/`fd_order` + domain + time → `SimulationSession`.
 */

#include <utility>

#include <mpi.h>

#include <openpfc/frontend/ui/from_json_session_selection.hpp>
#include <openpfc/frontend/ui/from_json_world_time.hpp>
#include <openpfc/kernel/simulation/simulation_session.hpp>

namespace pfc::ui {

template <class Stack>
[[nodiscard]] inline pfc::sim::SimulationSession<Stack>
make_simulation_session(const json &settings, int rank, int nproc,
                        MPI_Comm comm = MPI_COMM_WORLD) {
  auto selection = from_json<pfc::sim::SessionSelection>(settings);
  auto domain = from_json<pfc::Domain>(settings);
  auto time = from_json<pfc::Time>(settings);
  return pfc::sim::SimulationSession<Stack>(selection, std::move(domain),
                                            std::move(time), rank, nproc, comm);
}

} // namespace pfc::ui
