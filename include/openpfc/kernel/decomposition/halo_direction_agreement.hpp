// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file halo_direction_agreement.hpp
 * @brief Collective neighbour agreement check for HaloDirectionSet.
 *
 * @details
 * Exchanger constructors resolve the active direction set per rank via
 * `resolve_direction_set`. Paired ranks that share a face must agree on the
 * directions that cross that boundary (local `d` implies peer has `-d`),
 * otherwise MPI tags mismatch and the first Waitall can hang.
 *
 * `validate_neighbour_direction_agreement` Allgathers a canonical bitmask of
 * each rank's resolved set and checks every active direction on every rank
 * against its neighbour's opposite bit — fail-closed at construction.
 *
 * @see halo_directions.hpp
 * @see docs/concepts/halo_exchange.md §5
 */

#include <cstdint>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>

namespace pfc::halo {
namespace detail {

/// Inverse of `direction_to_canonical_tag` for tags in `[0, 33)`.
[[nodiscard]] inline HaloDirectionSet::Int3
canonical_tag_to_direction(int tag) {
  if (tag >= 0 && tag < 6) {
    return face_slot_to_direction(tag);
  }
  // Scan the 26 non-zero unit vectors; tags 6..32 encode edges/corners.
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        if (dx == 0 && dy == 0 && dz == 0) {
          continue;
        }
        const HaloDirectionSet::Int3 d{dx, dy, dz};
        if (direction_to_canonical_tag(d) == tag) {
          return d;
        }
      }
    }
  }
  throw std::runtime_error(
      "canonical_tag_to_direction: no direction for tag " + std::to_string(tag));
}

[[nodiscard]] inline std::uint64_t
encode_direction_set_mask(const HaloDirectionSet &dirs) {
  std::uint64_t mask = 0;
  for (const auto &d : dirs.dirs) {
    const int tag = direction_to_canonical_tag(d);
    mask |= (std::uint64_t{1} << tag);
  }
  return mask;
}

[[nodiscard]] inline std::string format_direction(const HaloDirectionSet::Int3 &d) {
  return "{" + std::to_string(d[0]) + "," + std::to_string(d[1]) + "," +
         std::to_string(d[2]) + "}";
}

} // namespace detail

/**
 * @brief Fail-closed collective check that neighbouring ranks agree on
 *        HaloDirectionSet directions across shared boundaries.
 *
 * @param comm   MPI communicator (same one the exchanger will use).
 * @param decomp Decomposition used to resolve neighbour ranks.
 * @param rank   Local MPI rank (included for API symmetry / diagnostics).
 * @param dirs   Resolved direction set for @p rank.
 *
 * @throws std::runtime_error when any rank's active direction `d` toward
 *         neighbour `n` is not matched by `-d` in `n`'s set. Every rank
 *         validates the full gathered table so all ranks throw together.
 */
inline void validate_neighbour_direction_agreement(
    MPI_Comm comm, const decomposition::Decomposition &decomp, int rank,
    const HaloDirectionSet &dirs) {
  int size = 0;
  pfc::mpi::throw_on_mpi_error(MPI_Comm_size(comm, &size), "MPI_Comm_size");

  const std::uint64_t local_mask = detail::encode_direction_set_mask(dirs);
  std::vector<std::uint64_t> all(static_cast<std::size_t>(size), 0);
  pfc::mpi::throw_on_mpi_error(
      MPI_Allgather(&local_mask, 1, MPI_UINT64_T, all.data(), 1, MPI_UINT64_T,
                    comm),
      "MPI_Allgather");

  // Every rank walks the full table so a mismatch throws on all ranks.
  for (int r = 0; r < size; ++r) {
    const std::uint64_t mask_r = all[static_cast<std::size_t>(r)];
    for (int tag = 0; tag < 33; ++tag) {
      if ((mask_r & (std::uint64_t{1} << tag)) == 0) {
        continue;
      }
      const HaloDirectionSet::Int3 d = detail::canonical_tag_to_direction(tag);
      const int n = decomposition::get_neighbor_rank(decomp, r, d);
      if (n < 0 || n >= size) {
        throw std::runtime_error(
            "validate_neighbour_direction_agreement: invalid neighbour rank " +
            std::to_string(n) + " for rank " + std::to_string(r) + " direction " +
            detail::format_direction(d) + " (detected by rank " +
            std::to_string(rank) + ")");
      }
      const HaloDirectionSet::Int3 opp{-d[0], -d[1], -d[2]};
      const int opp_tag = direction_to_canonical_tag(opp);
      const std::uint64_t mask_n = all[static_cast<std::size_t>(n)];
      if ((mask_n & (std::uint64_t{1} << opp_tag)) == 0) {
        throw std::runtime_error(
            "HaloDirectionSet neighbour disagreement: rank " +
            std::to_string(r) + " has direction " + detail::format_direction(d) +
            " toward neighbour rank " + std::to_string(n) +
            ", but neighbour lacks opposite " + detail::format_direction(opp) +
            " (detected by rank " + std::to_string(rank) + ")");
      }
    }
  }
}

} // namespace pfc::halo
