// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file grain_relabel.hpp
 * @brief Persistent identities of one field on the host (reference).
 *
 * `relabel` keeps object identities across updates of a single amplitude
 * field, so two objects that come into contact keep two identities:
 *
 *   1. previous labels on core cells are propagated through the core
 *      (reference `propagate`: nearest seed, smallest identity on ties);
 *   2. each connected component of core cells that no previous label
 *      reaches gets a new identity from `sequence`;
 *   3. the core labels are propagated into the support.
 *
 * Support cells that no core label reaches stay background. A core cell
 * must also be a support cell. With no previous labels, step 2 labels every
 * core component. Single slot; small-grid host reference like `propagate`.
 */

#include <cstdint>
#include <deque>
#include <span>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/grain/topology.hpp>
#include <openpfc/runtime/cpu/grain_tracking.hpp>

namespace pfc::grain::tracking::reference {

struct RelabelResult {
  std::vector<Id> labels;  ///< Support labels; background elsewhere.
  std::size_t created = 0; ///< Identities allocated by this call.
};

template <class Grid>
  requires std::same_as<Grid, Grid2D> || std::same_as<Grid, Grid3D>
inline RelabelResult relabel(Grid grid, std::span<const std::uint8_t> core,
                             std::span<const std::uint8_t> support,
                             std::span<const Id> previous,
                             IdentitySequence &sequence) {
  const auto cells = cell_count(grid);
  if (core.size() != cells || support.size() != cells ||
      (!previous.empty() && previous.size() != cells))
    throw std::invalid_argument(
        "relabel: mask or label size does not match the grid");
  for (std::size_t i = 0; i < cells; ++i)
    if (core[i] && !support[i])
      throw std::invalid_argument("relabel: core outside support");
  std::vector<Id> seeds(cells, background);
  if (!previous.empty())
    for (std::size_t i = 0; i < cells; ++i)
      if (core[i]) seeds[i] = previous[i];
  auto inner = propagate(grid, 1, core, std::span<const Id>(seeds));
  RelabelResult result;
  const int zr = pfc::grain::detail::depth_radius(grid, 1);
  std::deque<std::size_t> queue;
  for (std::size_t start = 0; start < cells; ++start) {
    if (!core[start] || inner.labels[start] != background) continue;
    const Id id = allocate_identity(sequence);
    if (id == background)
      throw std::overflow_error("relabel: identity sequence exhausted");
    ++result.created;
    inner.labels[start] = id;
    queue.push_back(start);
    while (!queue.empty()) {
      const auto c = queue.front();
      queue.pop_front();
      for (int dz = -zr; dz <= zr; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            if ((!dx && !dy && !dz) ||
                !pfc::grain::detail::stencil(grid, dx, dy, dz, 1))
              continue;
            std::size_t other = 0;
            if (pfc::grain::detail::offset(grid, c, dx, dy, dz, other) &&
                core[other] && inner.labels[other] == background) {
              inner.labels[other] = id;
              queue.push_back(other);
            }
          }
    }
  }
  result.labels =
      propagate(grid, 1, support, std::span<const Id>(inner.labels)).labels;
  return result;
}

} // namespace pfc::grain::tracking::reference
