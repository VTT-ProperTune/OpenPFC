// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <array>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/brick_split.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace pfc::decomposition {

namespace {

void validate_split_box_ordering(const std::vector<Box3i> &subs, const Int3 &grid) {
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
  for (const auto &box : subs) {
    for (int d = 0; d < 3; ++d) bnd[d].push_back(box.low[d]);
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
    const auto &lo = subs[static_cast<std::size_t>(r)].low;
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

} // namespace

Decomposition::Decomposition(const Domain &domain, const Int3 &grid)
    : m_grid{grid[0], grid[1], grid[2]}, m_local_boxes(), m_domain(domain) {
  const Box3i global = pfc::domain::index_box(domain);
  m_local_boxes = split_box(global, grid);
  validate_split_box_ordering(m_local_boxes, grid);
}

[[nodiscard]] Decomposition create(const Domain &domain, const Int3 &grid) {
  return Decomposition(domain, grid);
}

[[nodiscard]] Decomposition create(const Domain &domain, const int &nparts) {
  const Int3 &size = domain.size;
  const long long total_grid_points = static_cast<long long>(size[0]) *
                                      static_cast<long long>(size[1]) *
                                      static_cast<long long>(size[2]);
  if (nparts > total_grid_points) {
    throw std::invalid_argument("Cannot create decomposition with " +
                                std::to_string(nparts) +
                                " parts for a domain with only " +
                                std::to_string(total_grid_points) + " grid points");
  }

  const Int3 grid = min_surface_proc_grid(size, nparts);
  return create(domain, grid);
}

} // namespace pfc::decomposition
