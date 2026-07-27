// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulation_factories.hpp
 * @brief Small shared factories for tests (Domain + World + decomposition shortcuts)
 *
 * M1 migration: Prefer Domain for test helpers; construct World only where
 * Model/FFT seams require World&.
 */

#ifndef OPENPFC_TESTS_FIXTURES_SIMULATION_FACTORIES_HPP
#define OPENPFC_TESTS_FIXTURES_SIMULATION_FACTORIES_HPP

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>

namespace pfc::test {

/** @brief Uniform 8³ grid domain (common in unit tests). */
[[nodiscard]] inline Domain make_domain_cube_8() {
  return pfc::domain::create(GridSize({8, 8, 8}).to_vector3());
}

/** @brief Uniform 8³ grid world (common in unit tests, for Model/FFT seams). */
[[nodiscard]] inline World make_world_cube_8() {
  auto domain = make_domain_cube_8();
  const pfc::Int3 lower{0, 0, 0};
  const pfc::Int3 upper{7, 7, 7};
  return World(lower, upper, domain);
}

/** @brief Single-domain decomposition for @p world (one MPI rank owns all). */
[[nodiscard]] inline pfc::decomposition::Decomposition
make_serial_decomposition(const World &world) {
  return pfc::decomposition::create(world, 1);
}

/** @brief Single-domain decomposition for @p domain (one MPI rank owns all). */
[[nodiscard]] inline pfc::decomposition::Decomposition
make_serial_decomposition(const Domain &domain) {
  return pfc::decomposition::create(domain, 1);
}

} // namespace pfc::test

#endif // OPENPFC_TESTS_FIXTURES_SIMULATION_FACTORIES_HPP
