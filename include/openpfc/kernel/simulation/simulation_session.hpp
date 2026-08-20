// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file simulation_session.hpp
 * @brief Owns `SessionSelection` + `Time` + a method×backend `Stack` (M10).
 *
 * @details
 * Stacks are non-copyable / non-movable; this session is the same. Construct
 * in place (or as a prvalue). `stack_builder<Stack>` selects the factory.
 * GPU specializations live in `runtime/gpu/session_gpu_stack_factory.hpp`.
 */

#include <utility>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/simulation/session_selection.hpp>
#include <openpfc/kernel/simulation/session_stack_factory.hpp>
#include <openpfc/kernel/simulation/simulation_driver.hpp>
#include <openpfc/kernel/simulation/time.hpp>

namespace pfc::sim {

template <class Stack> class SimulationSession {
public:
  SimulationSession(const SimulationSession &) = delete;
  SimulationSession &operator=(const SimulationSession &) = delete;
  SimulationSession(SimulationSession &&) = delete;
  SimulationSession &operator=(SimulationSession &&) = delete;

  SimulationSession(SessionSelection selection, pfc::Domain domain, Time time,
                    int rank, int nproc, MPI_Comm comm = MPI_COMM_WORLD)
      : m_selection(selection), m_time(std::move(time)),
        m_stack(stack_builder<Stack>::make(selection, std::move(domain), rank, nproc,
                                           comm)) {}

  [[nodiscard]] const SessionSelection &selection() const noexcept {
    return m_selection;
  }
  [[nodiscard]] const char *stack_name() const noexcept {
    return stack_builder<Stack>::name;
  }

  [[nodiscard]] Time &time() noexcept { return m_time; }
  [[nodiscard]] const Time &time() const noexcept { return m_time; }

  [[nodiscard]] Stack &stack() noexcept { return m_stack; }
  [[nodiscard]] const Stack &stack() const noexcept { return m_stack; }

  template <class Step, class OnStart = NoopHook, class Apply = NoopHook,
            class OnSave = NoopHook>
  void run(Step &&step, OnStart &&on_start = {}, Apply &&apply = {},
           OnSave &&on_save = {}) {
    pfc::sim::run(m_time, std::forward<Step>(step), std::forward<OnStart>(on_start),
                  std::forward<Apply>(apply), std::forward<OnSave>(on_save));
  }

private:
  SessionSelection m_selection{};
  Time m_time;
  Stack m_stack;
};

} // namespace pfc::sim
