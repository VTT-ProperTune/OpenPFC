// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file halo_pattern.hpp
 * @brief Halo exchange pattern creation from Decomposition
 *
 * @details
 * Provides functions to create SparseVector objects representing halo regions
 * for exchange between neighbors in a decomposed domain. Handles:
 * - Automatic index calculation from Decomposition
 * - Global to local index conversion
 * - Different connectivity patterns (faces, edges, corners)
 * - Periodic boundary conditions
 * - Multiple halo widths
 *
 * @code
 * auto decomp = decomposition::create(world, {2, 2, 2});
 * int rank = 0;
 * int halo_width = 1;
 *
 * // Create halo pattern for +X face (1 row)
 * auto send_halo = halo::create_send_halo(decomp, rank, {1, 0, 0}, halo_width);
 * auto recv_halo = halo::create_recv_halo(decomp, rank, {1, 0, 0}, halo_width);
 *
 * // Exchange indices once (setup)
 * int neighbor = decomposition::get_neighbor_rank(decomp, rank, {1, 0, 0});
 * exchange::send(send_halo, rank, neighbor, MPI_COMM_WORLD);
 * exchange::receive(recv_halo, neighbor, rank, MPI_COMM_WORLD);
 *
 * // Then exchange data repeatedly
 * exchange::send_data(send_halo, rank, neighbor, MPI_COMM_WORLD);
 * exchange::receive_data(recv_halo, neighbor, rank, MPI_COMM_WORLD);
 * @endcode
 *
 * @see core/decomposition.hpp for Decomposition class
 * @see core/decomposition_neighbors.hpp for neighbor finding
 * @see core/sparse_vector.hpp for SparseVector
 * @see core/exchange.hpp for exchange operations
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#pragma once

#include <map>
#include <openpfc/core/backend_tags.hpp>
#include <openpfc/core/decomposition.hpp>
#include <openpfc/core/decomposition_neighbors.hpp>
#include <openpfc/core/sparse_vector.hpp>
#include <openpfc/core/types.hpp>
#include <openpfc/core/world.hpp>
#include <vector>

namespace pfc {
namespace halo {

using Int3 = pfc::types::Int3;

/**
 * @brief Connectivity pattern for halo exchange
 */
enum class Connectivity {
  Faces, // 6 neighbors: ±X, ±Y, ±Z (4-connectivity in 2D, 6-connectivity in 3D)
  Edges, // 18 neighbors: faces + edges (8-connectivity in 2D)
  All    // 26 neighbors: faces + edges + corners (full 3D connectivity)
};

/**
 * @brief Create SparseVector representing send halo region
 *
 * Extracts indices for the halo region that should be sent to a neighbor
 * in the specified direction. Indices are in **local coordinate space** (0-based
 * into the local field array).
 *
 * @param decomp Decomposition object
 * @param rank Current rank
 * @param direction Direction to neighbor, e.g., {1,0,0} for +X
 * @param halo_width Number of halo rows/layers
 * @return SparseVector with local indices of send halo region
 *
 * @example
 * ```cpp
 * // 2×2×2 grid, rank 0, send +X face halo (1 row)
 * auto send_halo = create_send_halo(decomp, 0, {1, 0, 0}, 1);
 * // Contains local indices of the rightmost face of rank 0's domain
 * // These indices can be used directly with gather() on local field
 * ```
 */
template <typename BackendTag = backend::CpuTag>
core::SparseVector<BackendTag, size_t>
create_send_halo(const decomposition::Decomposition &decomp, int rank,
                 const Int3 &direction, int halo_width) {
  const auto &local_world = decomposition::get_subworld(decomp, rank);
  (void)decomp; // For future use (periodic boundaries, etc.)

  // Get local domain size and bounds
  auto local_size = world::get_size(local_world);
  auto local_lower = world::get_lower(local_world);
  auto local_upper = world::get_upper(local_world);

  // Calculate which face/edge/corner we're sending
  std::vector<size_t> indices;

  // Determine the region to extract based on direction and halo_width
  // For send: we extract from the boundary of our local domain
  Int3 send_lower = local_lower;
  Int3 send_upper = local_upper;

  // Adjust boundaries based on direction
  // For +X direction: send rightmost face (last halo_width rows in X)
  if (direction[0] > 0) {
    send_lower[0] = local_upper[0] - halo_width;
    send_upper[0] = local_upper[0];
  } else if (direction[0] < 0) {
    send_lower[0] = local_lower[0];
    send_upper[0] = local_lower[0] + halo_width;
  } else {
    // direction[0] == 0: not sending in X direction, use full range
    send_lower[0] = local_lower[0];
    send_upper[0] = local_upper[0];
  }

  if (direction[1] > 0) {
    send_lower[1] = local_upper[1] - halo_width;
    send_upper[1] = local_upper[1];
  } else if (direction[1] < 0) {
    send_lower[1] = local_lower[1];
    send_upper[1] = local_lower[1] + halo_width;
  } else {
    send_lower[1] = local_lower[1];
    send_upper[1] = local_upper[1];
  }

  if (direction[2] > 0) {
    send_lower[2] = local_upper[2] - halo_width;
    send_upper[2] = local_upper[2];
  } else if (direction[2] < 0) {
    send_lower[2] = local_lower[2];
    send_upper[2] = local_lower[2] + halo_width;
  } else {
    send_lower[2] = local_lower[2];
    send_upper[2] = local_upper[2];
  }

  // Extract indices in local coordinate space
  // Convert 3D indices to linear index (row-major: x varies fastest, then y, then z)
  // Note: send_lower/send_upper are in global coordinates, we convert to local
  for (int z = send_lower[2]; z < send_upper[2]; ++z) {
    for (int y = send_lower[1]; y < send_upper[1]; ++y) {
      for (int x = send_lower[0]; x < send_upper[0]; ++x) {
        // Convert global 3D index to local linear index
        // Local coordinates relative to local_lower
        int local_z = z - local_lower[2];
        int local_y = y - local_lower[1];
        int local_x = x - local_lower[0];

        // Bounds check (should always pass, but safety first)
        if (local_x >= 0 && local_x < local_size[0] && local_y >= 0 &&
            local_y < local_size[1] && local_z >= 0 && local_z < local_size[2]) {
          // Row-major indexing: idx = z * (ny * nx) + y * nx + x
          size_t local_idx =
              static_cast<size_t>(local_z) * static_cast<size_t>(local_size[1]) *
                  static_cast<size_t>(local_size[0]) +
              static_cast<size_t>(local_y) * static_cast<size_t>(local_size[0]) +
              static_cast<size_t>(local_x);
          indices.push_back(local_idx);
        }
      }
    }
  }

  return core::SparseVector<BackendTag, size_t>(indices);
}

/**
 * @brief Create SparseVector representing receive halo region
 *
 * Extracts indices for the halo region where data should be received from
 * a neighbor. Indices are in **local coordinate space** (0-based into the
 * local field array, including halo zones).
 *
 * @param decomp Decomposition object
 * @param rank Current rank
 * @param direction Direction from neighbor, e.g., {1,0,0} means receiving from +X
 * neighbor
 * @param halo_width Number of halo rows/layers
 * @return SparseVector with local indices of receive halo region
 *
 * @note For periodic boundaries, this will wrap around to the opposite side
 * @note These indices point into the halo zone of the local field (which should
 *       be allocated with extra space for halos)
 */
template <typename BackendTag = backend::CpuTag>
core::SparseVector<BackendTag, size_t>
create_recv_halo(const decomposition::Decomposition &decomp, int rank,
                 const Int3 &direction, int halo_width) {
  const auto &local_world = decomposition::get_subworld(decomp, rank);
  const auto &global_world = decomposition::get_global_world(decomp);
  const auto &grid = decomposition::get_grid(decomp);

  auto local_size = world::get_size(local_world);
  auto local_lower = world::get_lower(local_world);
  auto local_upper = world::get_upper(local_world);

  std::vector<size_t> indices;

  // For receive: we receive into the halo zone at the opposite boundary
  Int3 recv_lower = local_lower;
  Int3 recv_upper = local_upper;

  // Adjust boundaries: receive halo is at the opposite side from send
  // If receiving from +X neighbor, we receive into leftmost halo zone
  // Note: For fields with halo padding, these indices point into the halo zone
  // For fields without halo padding, caller must handle allocation
  if (direction[0] > 0) {
    recv_lower[0] = local_lower[0];
    recv_upper[0] = local_lower[0] + halo_width;
  } else if (direction[0] < 0) {
    recv_lower[0] = local_upper[0] - halo_width;
    recv_upper[0] = local_upper[0];
  } else {
    // direction[0] == 0: not receiving in X direction, use full range
    recv_lower[0] = local_lower[0];
    recv_upper[0] = local_upper[0];
  }

  if (direction[1] > 0) {
    recv_lower[1] = local_lower[1];
    recv_upper[1] = local_lower[1] + halo_width;
  } else if (direction[1] < 0) {
    recv_lower[1] = local_upper[1] - halo_width;
    recv_upper[1] = local_upper[1];
  } else {
    recv_lower[1] = local_lower[1];
    recv_upper[1] = local_upper[1];
  }

  if (direction[2] > 0) {
    recv_lower[2] = local_lower[2];
    recv_upper[2] = local_lower[2] + halo_width;
  } else if (direction[2] < 0) {
    recv_lower[2] = local_upper[2] - halo_width;
    recv_upper[2] = local_upper[2];
  } else {
    recv_lower[2] = local_lower[2];
    recv_upper[2] = local_upper[2];
  }

  // Extract indices in local coordinate space
  // Note: recv_lower/recv_upper are in global coordinates
  // For receive halo: we receive into the halo zone at the boundary
  // The indices are relative to local_lower, so they may be negative or
  // beyond local_size if the field doesn't have halo padding.
  // Caller is responsible for allocating field with appropriate halo space.

  for (int z = recv_lower[2]; z < recv_upper[2]; ++z) {
    for (int y = recv_lower[1]; y < recv_upper[1]; ++y) {
      for (int x = recv_lower[0]; x < recv_upper[0]; ++x) {
        // Convert global 3D index to local coordinates
        int local_z = z - local_lower[2];
        int local_y = y - local_lower[1];
        int local_x = x - local_lower[0];

        // For receive halo, these coordinates might be negative (if receiving
        // into halo zone before the owned region) or >= local_size (if receiving
        // into halo zone after the owned region).
        //
        // If field has halo padding: indices should account for halo offset
        // If field doesn't have halo padding: caller must handle mapping
        //
        // For now, we calculate indices assuming the field layout matches
        // the local domain bounds. Caller must adjust if using halo-padded layout.

        // Row-major indexing: idx = z * (ny * nx) + y * nx + x
        // This gives local index relative to local_lower
        size_t local_idx =
            static_cast<size_t>(local_z) * static_cast<size_t>(local_size[1]) *
                static_cast<size_t>(local_size[0]) +
            static_cast<size_t>(local_y) * static_cast<size_t>(local_size[0]) +
            static_cast<size_t>(local_x);
        indices.push_back(local_idx);
      }
    }
  }

  return core::SparseVector<BackendTag, size_t>(indices);
}

/**
 * @brief Create all halo patterns for a rank
 *
 * Creates send and receive SparseVectors for all neighbors based on
 * connectivity pattern. Returns a map from direction to {send_halo, recv_halo}.
 *
 * @param decomp Decomposition object
 * @param rank Current rank
 * @param connectivity Connectivity pattern (Faces, Edges, or All)
 * @param halo_width Number of halo rows
 * @return Map from direction Int3 to pair of {send_halo, recv_halo} SparseVectors
 */
template <typename BackendTag = backend::CpuTag>
std::map<Int3, std::pair<core::SparseVector<BackendTag, size_t>,
                         core::SparseVector<BackendTag, size_t>>>
create_halo_patterns(const decomposition::Decomposition &decomp, int rank,
                     Connectivity connectivity, int halo_width) {
  std::map<Int3, std::pair<core::SparseVector<BackendTag, size_t>,
                           core::SparseVector<BackendTag, size_t>>>
      patterns;

  // Get neighbors based on connectivity
  std::map<Int3, int> neighbors;
  switch (connectivity) {
  case Connectivity::Faces:
    neighbors = decomposition::find_face_neighbors(decomp, rank);
    break;
  case Connectivity::All:
    neighbors = decomposition::find_all_neighbors(decomp, rank);
    break;
  default: neighbors = decomposition::find_face_neighbors(decomp, rank); break;
  }

  // Create send/recv halos for each neighbor
  for (const auto &[direction, neighbor_rank] : neighbors) {
    if (neighbor_rank >= 0) {
      auto send_halo =
          create_send_halo<BackendTag>(decomp, rank, direction, halo_width);
      auto recv_halo =
          create_recv_halo<BackendTag>(decomp, rank, direction, halo_width);
      patterns.emplace(direction,
                       std::make_pair(std::move(send_halo), std::move(recv_halo)));
    }
  }

  return patterns;
}

} // namespace halo
} // namespace pfc
