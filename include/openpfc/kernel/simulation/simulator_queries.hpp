// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulator_queries.hpp
 * @brief Free functions for `Simulator` accessors and default physics step
 *
 * @details
 * Included from `simulator.hpp` after `simulator_integrator.hpp`. Keeps the
 * main `Simulator` class header focused on the type surface; these helpers are
 * the stable non-member API used by examples and tests.
 *
 * @see simulator.hpp
 */

#ifndef PFC_KERNEL_SIMULATION_SIMULATOR_QUERIES_HPP
#define PFC_KERNEL_SIMULATION_SIMULATOR_QUERIES_HPP

#include <openpfc/kernel/simulation/model.hpp>

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

inline void step(Simulator &s, Model &m) {
  pfc::step(m, pfc::time::current(get_time(s)));
}

#endif // PFC_KERNEL_SIMULATION_SIMULATOR_QUERIES_HPP
