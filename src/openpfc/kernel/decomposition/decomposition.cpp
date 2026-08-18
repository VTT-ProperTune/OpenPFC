// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <array>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/brick_split.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace pfc::decomposition {

namespace {

// Fail closed (audit 4.9 / ADR 0007): get_neighbor_rank() and the halo
// machinery assume x-fastest rank order, rank = cz*gx*gy + cy*gx + cx.
void validate_split_world_ordering(const std::vector<World> &subs,
                                   const Int3 &grid) {
  const int gx = grid[0], gy = grid[1], gz = grid[2];
  const long long expected = static_cast<long long>(gx) * gy * gz;

  if (static_cast<long long>(subs.size()) != expected) {
    throw std::runtime_error(
        "Decomposition: in-repo splitter produced " + std::to_string(subs.size()) +
        " subdomains for a " + std::to_string(gx) + "x" + std::to_string(gy) + "x" +
        std::to_string(gz) + " process grid (expected " + std::to_string(expected) +
        ")");
  }

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
        "Decomposition: in-repo splitter partition is not a regular "
        "gx*gy*gz Cartesian grid; the x-fastest neighbor arithmetic in "
        "get_neighbor_rank would be invalid.");
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
          "Decomposition: in-repo splitter box ordering does not match the "
          "x-fastest rank convention used by get_neighbor_rank (subdomain " +
          std::to_string(r) + " sits at grid coordinate (" + std::to_string(cx) +
          "," + std::to_string(cy) + "," + std::to_string(cz) +
          ") which implies MPI_Cart_shift rank " + std::to_string(implied) + ")");
    }
  }
}

[[nodiscard]] std::vector<World> split_world_bricks(const World &world,
                                                    const Int3 &grid) {
  const Int3 lo = pfc::world::get_lower(world);
  const Int3 hi = pfc::world::get_upper(world);
  const Box3i global = Box3i::from_bounds(
      {lo[0], lo[1], lo[2]}, {hi[0], hi[1], hi[2]});
  std::vector<World> sub_worlds;
  for (const auto &box : split_box(global, grid)) {
    sub_worlds.push_back(World(Int3{box.low[0], box.low[1], box.low[2]},
                               Int3{box.high[0], box.high[1], box.high[2]},
                               pfc::world::get_coordinate_system(world)));
  }
  validate_split_world_ordering(sub_worlds, grid);
  return sub_worlds;
}

} // namespace

Decomposition::Decomposition(const World &world, const Int3 &grid)
    : m_global_world(world), m_grid{grid[0], grid[1], grid[2]},
      m_local_boxes(), m_domain() {
  const std::vector<World> subworlds = split_world_bricks(world, grid);

  m_local_boxes.reserve(subworlds.size());
  for (const auto &subworld : subworlds) {
    m_local_boxes.push_back(Box3i::from_bounds(
        pfc::world::get_lower(subworld), pfc::world::get_upper(subworld)));
  }

  m_domain = pfc::domain::create(
      pfc::GridSize(pfc::world::get_size(world)),
      pfc::PhysicalOrigin(pfc::world::get_origin(world)),
      pfc::GridSpacing(pfc::world::get_spacing(world)),
      pfc::world::get_periodic(world));
}

[[nodiscard]] Decomposition create(const World &world, const Int3 &grid) {
  return Decomposition(world, grid);
}

[[nodiscard]] Decomposition create(const World &world, const int &nparts) {
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

  const Int3 grid = min_surface_proc_grid(size, nparts);
  return create(world, grid);
}

[[nodiscard]] Decomposition create(const Domain &domain, const Int3 &grid) {
  const Int3 &size = domain.size;
  const Int3 lower{0, 0, 0};
  const Int3 upper{size[0] - 1, size[1] - 1, size[2] - 1};
  const World world(lower, upper, domain);
  return create(world, grid);
}

[[nodiscard]] Decomposition create(const Domain &domain, const int &nparts) {
  const Int3 &size = domain.size;
  const Int3 lower{0, 0, 0};
  const Int3 upper{size[0] - 1, size[1] - 1, size[2] - 1};
  const World world(lower, upper, domain);
  return create(world, nparts);
}

} // namespace pfc::decomposition
