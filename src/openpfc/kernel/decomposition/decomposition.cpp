// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <heffte.h>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>

namespace pfc::decomposition {

namespace {

[[nodiscard]] heffte::box3d<int> global_world_to_heffte_box(const World &world) {
  return heffte::box3d<int>(pfc::world::get_lower(world),
                            pfc::world::get_upper(world));
}

[[nodiscard]] World subworld_from_heffte_box(const World &global,
                                             const heffte::box3d<int> &box) {
  return World(box.low, box.high, pfc::world::get_coordinate_system(global));
}

[[nodiscard]] std::vector<World> split_world_heffte(const World &world,
                                                    const Int3 &grid) {
  const heffte::box3d<int> global_box = global_world_to_heffte_box(world);
  std::vector<World> sub_worlds;
  for (const auto &box : heffte::split_world(global_box, grid)) {
    sub_worlds.push_back(subworld_from_heffte_box(world, box));
  }
  return sub_worlds;
}

} // namespace

Decomposition::Decomposition(const World &world, const Int3 &grid)
    : m_global_world(world), m_grid{grid[0], grid[1], grid[2]},
      m_subworlds(split_world_heffte(world, grid)) {}

[[nodiscard]] Decomposition create(const World &world, const Int3 &grid) noexcept {
  return Decomposition(world, grid);
}

[[nodiscard]] Decomposition create(const World &world, const int &nparts) noexcept {
  const heffte::box3d<int> indices = global_world_to_heffte_box(world);
  const auto grid = heffte::proc_setup_min_surface(indices, nparts);
  return create(world, grid);
}

} // namespace pfc::decomposition
