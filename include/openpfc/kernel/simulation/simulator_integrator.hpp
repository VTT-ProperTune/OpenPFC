// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulator_integrator.hpp
 * @brief Time-step prologue/epilogue and scheduled results write for `Simulator`
 *
 * @details
 * Included from `simulator.hpp` immediately after the `Simulator` class definition.
 * Keeps orchestration helpers out of the main class header (SRP / readability).
 *
 * @see simulator.hpp
 * @see simulator_queries.hpp for `get_model` / `get_time` free functions
 */

#ifndef PFC_KERNEL_SIMULATION_SIMULATOR_INTEGRATOR_HPP
#define PFC_KERNEL_SIMULATION_SIMULATOR_INTEGRATOR_HPP

#include <openpfc/kernel/simulation/simulator_results_dispatch.hpp>
#include <openpfc/kernel/simulation/time.hpp>

/**
 * @brief One scheduled results write: dispatch writers then bump `result_counter`
 *
 * Same logic as `Simulator::write_results()`. Prefer this free function in
 * tests or tools that need a callable seam without member syntax.
 *
 * @see Simulator::write_results()
 * @see pfc::write_results_for_registered_fields
 */
inline void write_scheduled_simulator_results(Simulator &sim) {
  const int file_num = sim.get_result_counter();
  pfc::write_results_for_registered_fields(sim.get_model(), sim.results_writers(),
                                           file_num);
  sim.set_result_counter(file_num + 1);
}

inline void Simulator::write_results() { write_scheduled_simulator_results(*this); }

namespace simulator_integrator {

/** @brief Shared body of `Simulator::begin_integrator_step` (ordering contract). */
inline void begin_integrator_step(Simulator &sim) {
  Time &time = sim.get_time();
  if (pfc::time::increment(time) == 0) {
    sim.apply_initial_conditions();
    sim.apply_boundary_conditions();
    if (pfc::time::do_save(time)) {
      sim.write_results();
    }
  }
  pfc::time::next(time);
  sim.apply_boundary_conditions();
}

/** @brief Shared body of `Simulator::end_integrator_step`. */
inline void end_integrator_step(Simulator &sim) {
  if (pfc::time::do_save(sim.get_time())) {
    sim.write_results();
  }
}

} // namespace simulator_integrator

inline void Simulator::begin_integrator_step() {
  simulator_integrator::begin_integrator_step(*this);
}

inline void Simulator::end_integrator_step() {
  simulator_integrator::end_integrator_step(*this);
}

#endif // PFC_KERNEL_SIMULATION_SIMULATOR_INTEGRATOR_HPP
