// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <array>
#include <heffte.h>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <stdexcept>
#include <string>
#include <vector>

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

// Fail closed (audit 4.9): get_neighbor_rank() and the halo machinery assume
// heffte::split_world enumerates subdomain boxes in x-fastest rank order, i.e.
// rank = cz*gx*gy + cy*gx + cx, with (cx,cy,cz) the box's grid coordinate. This
// invariant is implicit; a HeFFTe change to the enumeration order would silently
// corrupt every halo exchange. Verify it at construction by deriving each box's
// grid coordinate from its (regular, Cartesian-product) lower bounds and
// checking it maps back to the box's index.
void validate_split_world_ordering(const std::vector<World> &subs,
                                   const Int3 &grid) {
  const int gx = grid[0], gy = grid[1], gz = grid[2];
  const long long expected = static_cast<long long>(gx) * gy * gz;
  if (static_cast<long long>(subs.size()) != expected) {
    throw std::runtime_error("Decomposition: heffte::split_world produced " +
                             std::to_string(subs.size()) + " subdomains for a " +
                             std::to_string(gx) + "x" + std::to_string(gy) + "x" +
                             std::to_string(gz) + " process grid (expected " +
                             std::to_string(expected) +
                             "). The installed HeFFTe version may be incompatible.");
  }

  // Distinct, sorted lower-bound values along each axis are the partition
  // boundaries; there must be exactly grid[d] of them.
  std::array<std::vector<int>, 3> bnd;
  for (const auto &w : subs) {
    const Int3 lo = pfc::world::get_lower(w);
    for (int d = 0; d < 3; ++d) bnd[d].push_back(lo[d]);
  }
  for (int d = 0; d < 3; ++d) {
    std::sort(bnd[d].begin(), bnd[d].end());
    bnd[d].erase(std::unique(bnd[d].begin(), bnd[d].end()), bnd[d].end());
  }
  if (static_cast<int>(bnd[0].size()) != gx ||
      static_cast<int>(bnd[1].size()) != gy ||
      static_cast<int>(bnd[2].size()) != gz) {
    throw std::runtime_error(
        "Decomposition: heffte::split_world partition is not a regular "
        "gx*gy*gz Cartesian grid; the x-fastest neighbor arithmetic in "
        "get_neighbor_rank would be invalid. The installed HeFFTe version may "
        "be incompatible.");
  }

  auto coord_of = [](const std::vector<int> &axis, int value) {
    return static_cast<int>(std::lower_bound(axis.begin(), axis.end(), value) -
                            axis.begin());
  };
  for (int r = 0; r < static_cast<int>(subs.size()); ++r) {
    const Int3 lo = pfc::world::get_lower(subs[r]);
    const int cx = coord_of(bnd[0], lo[0]);
    const int cy = coord_of(bnd[1], lo[1]);
    const int cz = coord_of(bnd[2], lo[2]);
    const int implied = cz * gx * gy + cy * gx + cx;
    if (implied != r) {
      throw std::runtime_error(
          "Decomposition: heffte::split_world box ordering does not match the "
          "x-fastest rank convention used by get_neighbor_rank (subdomain " +
          std::to_string(r) + " sits at grid coordinate (" + std::to_string(cx) +
          "," + std::to_string(cy) + "," + std::to_string(cz) +
          ") which implies rank " + std::to_string(implied) +
          "). Halo exchange would be corrupted; the installed HeFFTe version "
          "may be incompatible.");
    }
  }
}

[[nodiscard]] std::vector<World> split_world_heffte(const World &world,
                                                    const Int3 &grid) {
  const heffte::box3d<int> global_box = global_world_to_heffte_box(world);
  std::vector<World> sub_worlds;
  for (const auto &box : heffte::split_world(global_box, grid)) {
    sub_worlds.push_back(subworld_from_heffte_box(world, box));
  }
  validate_split_world_ordering(sub_worlds, grid);
  return sub_worlds;
}

} // namespace

Decomposition::Decomposition(const World &world, const Int3 &grid)
    : m_global_world(world), m_grid{grid[0], grid[1], grid[2]},
      m_subworlds(split_world_heffte(world, grid)) {}

[[nodiscard]] Decomposition create(const World &world, const Int3 &grid) {
  return Decomposition(world, grid);
}

[[nodiscard]] Decomposition create(const World &world, const int &nparts) {
  // Validate nparts against total grid points before calling HeFFTe
  const Int3 size = pfc::world::get_size(world);
  const long long total_grid_points = static_cast<long long>(size[0]) *
                                      static_cast<long long>(size[1]) *
                                      static_cast<long long>(size[2]);
  if (nparts > total_grid_points) {
    throw std::invalid_argument("Cannot create decomposition with " +
                                std::to_string(nparts) +
                                " parts for a world with only " +
                                std::to_string(total_grid_points) + " grid points");
  }

  const heffte::box3d<int> indices = global_world_to_heffte_box(world);
  const auto grid = heffte::proc_setup_min_surface(indices, nparts);
  return create(world, grid);
}

} // namespace pfc::decomposition
