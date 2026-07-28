// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulation_factories.hpp
 * @brief Small shared factories for tests (domain + decomposition shortcuts)
 */

#ifndef OPENPFC_TESTS_FIXTURES_SIMULATION_FACTORIES_HPP
#define OPENPFC_TESTS_FIXTURES_SIMULATION_FACTORIES_HPP

#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>

namespace pfc::test {

/** @brief Uniform 8³ grid world (common in unit tests). */
[[nodiscard]] inline World make_world_cube_8() {
  using namespace pfc;
  auto domain = domain::create({8, 8, 8});
  return World({0, 0, 0}, {7, 7, 7}, domain);
}

/** @brief Single-domain decomposition for @p domain (one MPI rank owns all). */
[[nodiscard]] inline pfc::decomposition::Decomposition
make_serial_decomposition(const Domain &domain) {
  return pfc::decomposition::create(domain, 1);
}

/** @brief Single-domain decomposition for @p world (one MPI rank owns all). */
[[nodiscard]] inline pfc::decomposition::Decomposition
make_serial_decomposition(const World &world) {
  using namespace pfc;
  auto domain = world::get_coordinate_system(world);
  return pfc::decomposition::create(domain, 1);
}

} // namespace pfc::test

#endif // OPENPFC_TESTS_FIXTURES_SIMULATION_FACTORIES_HPP
