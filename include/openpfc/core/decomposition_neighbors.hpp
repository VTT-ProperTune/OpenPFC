// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file decomposition_neighbors.hpp
 * @brief Neighbor finding utilities for Decomposition
 *
 * @details
 * Provides functions to determine neighbor ranks in a 3D grid decomposition.
 * Used for halo exchange patterns in finite difference methods.
 *
 * @code
 * auto decomp = decomposition::create(world, {2, 2, 2}); // 2×2×2 grid
 * int rank = 0;
 *
 * // Find neighbor in +X direction (with periodic boundaries)
 * auto neighbor = decomposition::get_neighbor_rank(decomp, rank, {1, 0, 0});
 * // Always returns a valid neighbor rank (wraps around at boundaries)
 *
 * // Find all 6 face neighbors (always 6 with periodic boundaries)
 * auto neighbors = decomposition::find_face_neighbors(decomp, rank);
 * // Returns map: direction -> neighbor_rank (always 6 entries)
 * @endcode
 *
 * @see core/decomposition.hpp for Decomposition class
 * @see core/exchange.hpp for SparseVector exchange operations
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#pragma once

#include <map>
#include <openpfc/core/decomposition.hpp>
#include <openpfc/core/types.hpp>
#include <optional>

namespace pfc {
namespace decomposition {

/**
 * @brief Get neighbor rank in a given direction
 *
 * Given a decomposition grid and current rank, returns the rank of the neighbor
 * in the specified direction. With periodic boundary conditions, every rank
 * has neighbors in all directions (wraps around the domain).
 *
 * @param decomp Decomposition object
 * @param rank Current rank (0 to num_domains-1)
 * @param direction Direction vector, e.g., {1,0,0} for +X, {-1,0,0} for -X
 * @return Neighbor rank (always valid with periodic boundaries)
 *
 * @example
 * ```cpp
 * auto decomp = decomposition::create(world, {2, 2, 2}); // 8 ranks in 2×2×2 grid
 *
 * // Rank 0 in 2×2×2 grid: neighbors are 1 (+X), 2 (+Y), 4 (+Z)
 * int neighbor_x = get_neighbor_rank(decomp, 0, {1, 0, 0}); // Returns 1
 * int neighbor_y = get_neighbor_rank(decomp, 0, {0, 1, 0}); // Returns 2
 * int neighbor_z = get_neighbor_rank(decomp, 0, {0, 0, 1}); // Returns 4
 * int neighbor_nx = get_neighbor_rank(decomp, 0, {-1, 0, 0}); // Returns 1 (wraps to
 * rightmost)
 * ```
 */
inline int get_neighbor_rank(const Decomposition &decomp, int rank,
                             const Int3 &direction) {
  const auto &grid = get_grid(decomp);
  int num_domains = get_num_domains(decomp);

  if (rank < 0 || rank >= num_domains) {
    return -1; // Invalid rank
  }

  // Convert rank to 3D grid coordinates
  int rank_z = rank / (grid[0] * grid[1]);
  int rank_y = (rank % (grid[0] * grid[1])) / grid[0];
  int rank_x = rank % grid[0];

  // Calculate neighbor coordinates
  int neighbor_x = rank_x + direction[0];
  int neighbor_y = rank_y + direction[1];
  int neighbor_z = rank_z + direction[2];

  // Wrap around for periodic boundary conditions
  // This ensures every rank has neighbors in all directions
  neighbor_x = (neighbor_x + grid[0]) % grid[0];
  neighbor_y = (neighbor_y + grid[1]) % grid[1];
  neighbor_z = (neighbor_z + grid[2]) % grid[2];

  // Convert neighbor coordinates back to rank
  int neighbor_rank =
      neighbor_z * (grid[0] * grid[1]) + neighbor_y * grid[0] + neighbor_x;

  return neighbor_rank;
}

/**
 * @brief Find all face neighbors (6 directions: ±X, ±Y, ±Z)
 *
 * Returns a map of direction vectors to neighbor ranks. With periodic
 * boundary conditions, all 6 face neighbors are always present.
 *
 * @param decomp Decomposition object
 * @param rank Current rank
 * @return Map from direction Int3 to neighbor rank (always contains 6 entries)
 *
 * @example
 * ```cpp
 * auto decomp = decomposition::create(world, {2, 2, 2});
 * auto neighbors = find_face_neighbors(decomp, 0);
 *
 * // neighbors[{1,0,0}] = 1  (neighbor in +X)
 * // neighbors[{-1,0,0}] = 1 (neighbor in -X, wraps around)
 * // neighbors[{0,1,0}] = 2  (neighbor in +Y)
 * // neighbors[{0,-1,0}] = 0 (neighbor in -Y, wraps around)
 * // neighbors[{0,0,1}] = 4  (neighbor in +Z)
 * // neighbors[{0,0,-1}] = 0 (neighbor in -Z, wraps around)
 * ```
 */
inline std::map<Int3, int> find_face_neighbors(const Decomposition &decomp,
                                               int rank) {
  std::map<Int3, int> neighbors;

  // 6 face directions: ±X, ±Y, ±Z
  std::vector<Int3> directions = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                  {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};

  for (const auto &dir : directions) {
    int neighbor_rank = get_neighbor_rank(decomp, rank, dir);
    // With periodic boundaries, neighbor_rank is always valid (>= 0)
    neighbors[dir] = neighbor_rank;
  }

  return neighbors;
}

/**
 * @brief Find all neighbors (faces, edges, corners - 26 in 3D)
 *
 * Returns a map of all neighbor directions to ranks. With periodic
 * boundary conditions, all 26 neighbors are always present. Includes:
 * - 6 faces: ±X, ±Y, ±Z
 * - 12 edges: combinations like {1,1,0}, {1,-1,0}, etc.
 * - 8 corners: all combinations like {1,1,1}, {1,1,-1}, etc.
 *
 * @param decomp Decomposition object
 * @param rank Current rank
 * @return Map from direction Int3 to neighbor rank (always contains 26 entries)
 *
 * @example
 * ```cpp
 * auto decomp = decomposition::create(world, {3, 3, 3});
 * auto all_neighbors = find_all_neighbors(decomp, 13); // Center rank
 *
 * // Returns all 26 neighbors (faces + edges + corners)
 * // With periodic boundaries, even boundary ranks have all 26 neighbors
 * ```
 */
inline std::map<Int3, int> find_all_neighbors(const Decomposition &decomp,
                                              int rank) {
  std::map<Int3, int> neighbors;

  // All 26 directions in 3D: {-1,0,1} × {-1,0,1} × {-1,0,1} excluding {0,0,0}
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        if (dx == 0 && dy == 0 && dz == 0) {
          continue; // Skip self
        }

        Int3 direction{dx, dy, dz};
        int neighbor_rank = get_neighbor_rank(decomp, rank, direction);
        // With periodic boundaries, neighbor_rank is always valid (>= 0)
        neighbors[direction] = neighbor_rank;
      }
    }
  }

  return neighbors;
}

} // namespace decomposition
} // namespace pfc
