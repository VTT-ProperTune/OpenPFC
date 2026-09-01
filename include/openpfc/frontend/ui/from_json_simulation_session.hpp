// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file from_json_simulation_session.hpp
 * @brief JSON `method`/`backend`/`fd_order` + domain + time → `SimulationSession`.
 *
 * Spectral CPU stacks overlay HeFFTe `plan_options` from the same JSON
 * (`cpu_spectral_plan_options_from_json`). Other stacks use `stack_builder`.
 */

#include <type_traits>
#include <utility>

#include <mpi.h>

#include <openpfc/frontend/ui/from_json_session_selection.hpp>
#include <openpfc/frontend/ui/from_json_world_time.hpp>
#include <openpfc/frontend/ui/spectral_fft_stack_factory.hpp>
#include <openpfc/kernel/simulation/session_selection.hpp>
#include <openpfc/kernel/simulation/simulation_session.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

namespace pfc::ui {

template <class Stack>
[[nodiscard]] inline pfc::sim::SimulationSession<Stack>
make_simulation_session(const json &settings, int rank, int nproc,
                        MPI_Comm comm = MPI_COMM_WORLD) {
  auto selection = from_json<pfc::sim::SessionSelection>(settings);
  auto domain = from_json<pfc::Domain>(settings);
  auto time = from_json<pfc::Time>(settings);
  if constexpr (std::is_same_v<Stack, pfc::sim::stacks::SpectralCPUStack>) {
    pfc::sim::require_session_for_stack(selection,
                                        pfc::sim::SimulationMethod::Spectral,
                                        pfc::sim::SimulationBackend::Cpu);
    const auto options = cpu_spectral_plan_options_from_json(settings);
    return pfc::sim::SimulationSession<Stack>(selection, std::move(time), [&] {
      return pfc::sim::stacks::SpectralCPUStack(std::move(domain), rank, nproc, comm,
                                                options);
    });
  } else {
    return pfc::sim::SimulationSession<Stack>(selection, std::move(domain),
                                              std::move(time), rank, nproc, comm);
  }
}

} // namespace pfc::ui
