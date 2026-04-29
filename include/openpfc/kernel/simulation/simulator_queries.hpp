// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulator_queries.hpp
 * @brief Free functions for `Simulator` accessors, lifecycle, and default physics
 * step
 *
 * @details
 * Included from `simulator.hpp` after `simulator_integrator.hpp`. Keeps the
 * main `Simulator` class header focused on the type surface; these helpers are
 * the stable non-member API used by examples and tests.
 *
 * Prefer `pfc::initialize(sim)`, `pfc::step(sim)`, `pfc::done(sim)`, and the
 * integrator seam (`begin_integrator_step` / `end_integrator_step`) over member
 * spellings for consistency with `get_model` / `get_time` / `pfc::step(sim, model)`.
 *
 * @see simulator.hpp
 */

#ifndef PFC_KERNEL_SIMULATION_SIMULATOR_QUERIES_HPP
#define PFC_KERNEL_SIMULATION_SIMULATOR_QUERIES_HPP

#include <openpfc/kernel/simulation/model.hpp>
#include <openpfc/kernel/simulation/simulator_integrator.hpp>

[[nodiscard]] inline Model &get_model(Simulator &sim) noexcept {
  return sim.get_model();
}

[[nodiscard]] inline Time &get_time(Simulator &sim) noexcept {
  return sim.get_time();
}

[[nodiscard]] inline const World &get_world(Simulator &sim) noexcept {
  return pfc::get_world(get_model(sim));
}

[[nodiscard]] inline fft::IFFT &get_fft(Simulator &sim) noexcept {
  return pfc::get_fft(get_model(sim));
}

[[nodiscard]] inline bool is_rank0(const Simulator &sim) noexcept {
  return sim.is_rank0();
}

/** @brief Initialize the wrapped model with the simulator time step (`dt`). */
inline void initialize(Simulator &sim) { sim.initialize(); }

/** @brief One full orchestrated timestep (same as `Simulator::step()`). */
inline void step(Simulator &sim) { sim.step(); }

/** @brief True when simulation time has reached the end (same as
 * `Simulator::done()`). */
[[nodiscard]] inline bool done(Simulator &sim) { return sim.done(); }

/** @brief Integrator prologue (same as `Simulator::begin_integrator_step()`). */
inline void begin_integrator_step(Simulator &sim) {
  simulator_integrator::begin_integrator_step(sim);
}

/** @brief Integrator epilogue (same as `Simulator::end_integrator_step()`). */
inline void end_integrator_step(Simulator &sim) {
  simulator_integrator::end_integrator_step(sim);
}

/** @brief Completed physics steps after last `step()` (same as
 * `Simulator::get_increment()`). */
[[nodiscard]] inline unsigned int get_increment(Simulator &sim) {
  return sim.get_increment();
}

/** @brief Advance only the model with the simulator's current time (orchestration is
 * separate). */
inline void step(Simulator &s, Model &m) {
  pfc::step(m, pfc::time::current(get_time(s)));
}

#endif // PFC_KERNEL_SIMULATION_SIMULATOR_QUERIES_HPP
