// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
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
 * @see kernel/decomposition/decomposition.hpp for Decomposition class
 * @see kernel/decomposition/decomposition_neighbors.hpp for neighbor finding
 * @see kernel/decomposition/sparse_vector.hpp for SparseVector
 * @see kernel/decomposition/exchange.hpp for exchange operations
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#pragma once

#include <cstdint>
#include <map>
#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>
#include <openpfc/kernel/decomposition/sparse_vector.hpp>
#include <openpfc/kernel/execution/backend_tags.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace pfc::halo {

using Int3 = pfc::types::Int3;

namespace detail {

/// Fail closed when a direction-active owned axis cannot host `halo_width`.
/// Inactive axes (direction component 0) are not checked — flat Axes2D /
/// `nz==1` remain valid for face-only ±X/±Y exchanges (same spirit as
/// `create_face_types_6` after #204).
inline void ensure_halo_width_fits_direction(const Int3 &local_size,
                                             const Int3 &direction, int halo_width) {
  if (halo_width < 0) {
    throw std::invalid_argument("pfc::halo: halo_width must be non-negative (got " +
                                std::to_string(halo_width) + ")");
  }
  if (halo_width == 0) {
    return;
  }
  const char *axis_name[3] = {"x", "y", "z"};
  for (int a = 0; a < 3; ++a) {
    if (direction[a] == 0) {
      continue;
    }
    if (local_size[a] < halo_width) {
      throw std::invalid_argument(
          std::string("pfc::halo: local extent ") + std::to_string(local_size[0]) +
          "x" + std::to_string(local_size[1]) + "x" + std::to_string(local_size[2]) +
          " cannot host halo_width=" + std::to_string(halo_width) +
          " on direction (" + std::to_string(direction[0]) + "," +
          std::to_string(direction[1]) + "," + std::to_string(direction[2]) +
          ") (need >=" + std::to_string(halo_width) + " points on active " +
          axis_name[a] + " axis)");
    }
  }
}

} // namespace detail

/**
 * @brief Connectivity pattern for halo exchange
 */
enum class Connectivity : std::uint8_t {
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
  const auto owned_box = decomposition::local_box(decomp, rank);

  // Get local domain size and bounds
  auto local_size = owned_box.size;
  detail::ensure_halo_width_fits_direction(local_size, direction, halo_width);
  auto local_lower = owned_box.low;
  auto local_upper = owned_box.high;

  // Calculate which face/edge/corner we're sending
  std::vector<size_t> indices;

  // Determine the region to extract based on direction and halo_width
  // For send: we extract from the boundary of our local domain
  Int3 send_lower = local_lower;
  Int3 send_upper = local_upper;

  // Adjust boundaries based on direction
  // Loops use exclusive end: coord in [lo, hi_excl). World::get_upper() is
  // inclusive, so hi_excl for the owned box is local_upper + 1. For a positive
  // face (e.g. +X), the last halo_width owned cells are
  // [local_upper - halo_width + 1, local_upper] inclusive →
  // [local_upper - halo_width + 1, local_upper + 1) exclusive.
  if (direction[0] > 0) {
    send_lower[0] = local_upper[0] - halo_width + 1;
    send_upper[0] = local_upper[0] + 1;
  } else if (direction[0] < 0) {
    send_lower[0] = local_lower[0];
    send_upper[0] = local_lower[0] + halo_width;
  } else {
    // direction[0] == 0: not sending in X direction, use full range
    send_lower[0] = local_lower[0];
    send_upper[0] = local_upper[0] + 1;
  }

  if (direction[1] > 0) {
    send_lower[1] = local_upper[1] - halo_width + 1;
    send_upper[1] = local_upper[1] + 1;
  } else if (direction[1] < 0) {
    send_lower[1] = local_lower[1];
    send_upper[1] = local_lower[1] + halo_width;
  } else {
    send_lower[1] = local_lower[1];
    send_upper[1] = local_upper[1] + 1;
  }

  if (direction[2] > 0) {
    send_lower[2] = local_upper[2] - halo_width + 1;
    send_upper[2] = local_upper[2] + 1;
  } else if (direction[2] < 0) {
    send_lower[2] = local_lower[2];
    send_upper[2] = local_lower[2] + halo_width;
  } else {
    send_lower[2] = local_lower[2];
    send_upper[2] = local_upper[2] + 1;
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
 * @note These indices target **in-place** storage: boundary slabs of the same
 *       `nx×ny×nz` core array (`HaloExchanger`). For **separated** face buffers,
 *       use `pfc::SparseHaloExchanger` (typically built via
 *       `pfc::halo::make_structured_halos`) plus
 *       `pfc::halo::copy_to_face_layout` instead of scattering into these
 *       indices.
 */
template <typename BackendTag = backend::CpuTag>
core::SparseVector<BackendTag, size_t>
create_recv_halo(const decomposition::Decomposition &decomp, int rank,
                 const Int3 &direction, int halo_width) {
  const auto owned_box = decomposition::local_box(decomp, rank);
  [[maybe_unused]] const auto &global_world =
      decomposition::get_global_world(decomp);
  [[maybe_unused]] const auto &grid = decomposition::get_grid(decomp);

  auto local_size = owned_box.size;
  detail::ensure_halo_width_fits_direction(local_size, direction, halo_width);
  auto local_lower = owned_box.low;
  auto local_upper = owned_box.high;

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
    recv_lower[0] = local_upper[0] - halo_width + 1;
    recv_upper[0] = local_upper[0] + 1;
  } else {
    // direction[0] == 0: not receiving in X direction, use full range
    recv_lower[0] = local_lower[0];
    recv_upper[0] = local_upper[0] + 1;
  }

  if (direction[1] > 0) {
    recv_lower[1] = local_lower[1];
    recv_upper[1] = local_lower[1] + halo_width;
  } else if (direction[1] < 0) {
    recv_lower[1] = local_upper[1] - halo_width + 1;
    recv_upper[1] = local_upper[1] + 1;
  } else {
    recv_lower[1] = local_lower[1];
    recv_upper[1] = local_upper[1] + 1;
  }

  if (direction[2] > 0) {
    recv_lower[2] = local_lower[2];
    recv_upper[2] = local_lower[2] + halo_width;
  } else if (direction[2] < 0) {
    recv_lower[2] = local_upper[2] - halo_width + 1;
    recv_upper[2] = local_upper[2] + 1;
  } else {
    recv_lower[2] = local_lower[2];
    recv_upper[2] = local_upper[2] + 1;
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

        // In-place recv targets boundary slabs inside [0, nx·ny·nz). Fail
        // closed rather than casting negative / OOB coords to size_t.
        if (local_x < 0 || local_x >= local_size[0] || local_y < 0 ||
            local_y >= local_size[1] || local_z < 0 || local_z >= local_size[2]) {
          throw std::invalid_argument(
              "pfc::halo::create_recv_halo: local coordinate (" +
              std::to_string(local_x) + "," + std::to_string(local_y) + "," +
              std::to_string(local_z) + ") outside owned extents " +
              std::to_string(local_size[0]) + "x" + std::to_string(local_size[1]) +
              "x" + std::to_string(local_size[2]) + " for direction (" +
              std::to_string(direction[0]) + "," + std::to_string(direction[1]) +
              "," + std::to_string(direction[2]) +
              "), halo_width=" + std::to_string(halo_width));
        }
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

  // Get neighbors based on connectivity.
  //
  // Note: `Connectivity::Edges` previously fell into the default case and
  // silently aliased `Faces`. It now correctly returns the faces+edges
  // subset of the 26-direction enumeration (filters out the 8 corner
  // directions whose three components are all non-zero).
  std::map<Int3, int> neighbors;
  switch (connectivity) {
  case Connectivity::Faces:
    neighbors = decomposition::find_face_neighbors(decomp, rank);
    break;
  case Connectivity::Edges: {
    auto all = decomposition::find_all_neighbors(decomp, rank);
    for (const auto &[dir, nb] : all) {
      const int nz_components =
          (dir[0] != 0 ? 1 : 0) + (dir[1] != 0 ? 1 : 0) + (dir[2] != 0 ? 1 : 0);
      // Faces have exactly one non-zero component; edges have exactly two.
      // Corners (three non-zero components) are excluded.
      if (nz_components == 1 || nz_components == 2) {
        neighbors[dir] = nb;
      }
    }
    break;
  }
  case Connectivity::All:
    neighbors = decomposition::find_all_neighbors(decomp, rank);
    break;
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

} // namespace pfc::halo
