// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file halo_geometry.hpp
 * @brief Shared halo geometry: face slots, opposite neighbours, MPI tag
 *        blocks, and padded send/recv slabs (M4).
 *
 * @details
 * Several exchangers previously re-implemented `opposite_slot`, hand-spaced
 * MPI tags, and padded face/edge/corner slabs. This header is the single
 * source for those facts:
 *
 *   - 6-face slot order `+X,-X,+Y,-Y,+Z,-Z` (`kFaceDirections`)
 *   - `opposite_slot(slot) == slot ^ 1` and `opposite_direction(d)`
 *   - `direction_to_canonical_tag` in `[0, kCanonicalTagCount)`
 *   - per-field tag blocks of width `kCanonicalTagCount`, so two exchangers
 *     with distinct bases (or six fields in one exchanger) cannot collide
 *   - `padded_send_slab` / `padded_recv_slab` for any of the 26 neighbour
 *     directions on a halo-padded brick
 *
 * No MPI types and no Decomposition live here — those stay in the exchanger
 * headers. Device pack/unpack (CUDA/HIP) should consume these slabs rather
 * than restating the offsets. CUDA execution of those consumers is not
 * available on LUMI; HIP can be checked here, CUDA on tohtori.
 *
 * @see halo_directions.hpp for `HaloDirectionSet` presets
 * @see padded_halo_mpi_types.hpp for MPI subarrays built from these slabs
 */

#include <array>
#include <stdexcept>
#include <string>

#include <openpfc/kernel/data/types.hpp>

namespace pfc::halo {

using Int3 = pfc::types::Int3;

/// Canonical 6-face slot count (`+X,-X,+Y,-Y,+Z,-Z`).
inline constexpr int kFaceSlotCount = 6;

/// Width of one field's tag block (`direction_to_canonical_tag` range).
inline constexpr int kCanonicalTagCount = 33;

/// Face directions in slot order: 0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z.
inline constexpr std::array<Int3, kFaceSlotCount> kFaceDirections = {{
    {1, 0, 0},
    {-1, 0, 0},
    {0, 1, 0},
    {0, -1, 0},
    {0, 0, 1},
    {0, 0, -1},
}};

/**
 * @brief Opposite 6-face slot (`0↔1`, `2↔3`, `4↔5`).
 *
 * Equals `slot ^ 1` for `slot` in `[0, 6)`. Out-of-range slots are not
 * checked — callers that need a hard error use `face_slot_to_direction`.
 */
[[nodiscard]] constexpr int opposite_slot(int slot) noexcept { return slot ^ 1; }

/// Neighbour direction with every component negated.
[[nodiscard]] constexpr Int3 opposite_direction(const Int3 &d) noexcept {
  return Int3{-d[0], -d[1], -d[2]};
}

/**
 * @brief Map an axis-aligned direction to the 6-face slot index.
 *
 * @return Slot in `[0, 6)`, or `-1` for an edge/corner (or the zero vector).
 */
[[nodiscard]] inline int direction_to_face_slot(const Int3 &d) noexcept {
  for (int i = 0; i < kFaceSlotCount; ++i) {
    if (kFaceDirections[static_cast<std::size_t>(i)] == d) {
      return i;
    }
  }
  return -1;
}

/**
 * @brief Inverse of `direction_to_face_slot`.
 *
 * @throws std::out_of_range when `slot` is not in `[0, 6)`.
 */
[[nodiscard]] inline Int3 face_slot_to_direction(int slot) {
  if (slot < 0 || slot >= kFaceSlotCount) {
    throw std::out_of_range(
        "face_slot_to_direction: slot must be in [0,6), got " +
        std::to_string(slot));
  }
  return kFaceDirections[static_cast<std::size_t>(slot)];
}

/**
 * @brief Deterministic, peer-independent tag offset for any 26-direction
 *        vector, in `[0, kCanonicalTagCount)`.
 *
 * Faces use the 6-slot index. Edges and corners use
 * `6 + (dx+1) + 3*(dy+1) + 9*(dz+1)`. The encoding depends only on `d`,
 * so `send_tag(..., d)` on one rank equals `recv_tag(..., -d)` on the peer.
 */
[[nodiscard]] inline int direction_to_canonical_tag(const Int3 &d) noexcept {
  const int slot = direction_to_face_slot(d);
  if (slot >= 0) {
    return slot;
  }
  return 6 + (d[0] + 1) + 3 * (d[1] + 1) + 9 * (d[2] + 1);
}

/// First MPI tag of `field_index` inside an exchanger whose block starts at
/// `exchange_base`.
[[nodiscard]] constexpr int field_tag_base(int exchange_base,
                                           int field_index) noexcept {
  return exchange_base + field_index * kCanonicalTagCount;
}

/// Send tag for field `field_index` in direction `d`.
[[nodiscard]] inline int send_tag(int exchange_base, int field_index,
                                  const Int3 &d) noexcept {
  return field_tag_base(exchange_base, field_index) +
         direction_to_canonical_tag(d);
}

/// Recv tag for the same field/direction (peer's send of `-d`).
[[nodiscard]] inline int recv_tag(int exchange_base, int field_index,
                                  const Int3 &d) noexcept {
  return send_tag(exchange_base, field_index, opposite_direction(d));
}

/**
 * @brief Inclusive-start / exclusive-size subarray of a padded brick.
 *
 * Coordinates are in the padded index space of a field with owned extents
 * `owned` and halo width `hw` (outer size `owned + 2*hw` per axis).
 */
struct HaloSlab {
  Int3 start{};
  Int3 count{};

  [[nodiscard]] constexpr std::size_t volume() const noexcept {
    return static_cast<std::size_t>(count[0]) *
           static_cast<std::size_t>(count[1]) *
           static_cast<std::size_t>(count[2]);
  }
};

inline void validate_padded_slab_args(const Int3 &owned, int hw, const Int3 &d) {
  if (hw < 0) {
    throw std::invalid_argument(
        "halo slab: halo width must be non-negative (got " +
        std::to_string(hw) + ")");
  }
  if (d[0] == 0 && d[1] == 0 && d[2] == 0) {
    throw std::invalid_argument("halo slab: zero direction is not a neighbour");
  }
  for (int a = 0; a < 3; ++a) {
    if (d[a] < -1 || d[a] > 1) {
      throw std::invalid_argument(
          "halo slab: direction component must be in {-1,0,1}");
    }
    if (owned[a] < 0) {
      throw std::invalid_argument("halo slab: owned extent must be non-negative");
    }
    if (hw > 0 && d[a] != 0 && owned[a] < hw) {
      throw std::invalid_argument(
          "halo slab: owned extent on an active axis must be >= halo width");
    }
  }
}

/**
 * @brief Send slab: the `hw`-thick owned cells facing `d`.
 *
 * Inactive axes span the owned core (corners/edges are separate directions).
 * `hw == 0` yields a zero-count slab on every active axis.
 */
[[nodiscard]] inline HaloSlab padded_send_slab(const Int3 &owned, int hw,
                                               const Int3 &d) {
  validate_padded_slab_args(owned, hw, d);
  HaloSlab s{};
  for (int a = 0; a < 3; ++a) {
    const int lo = hw;
    if (d[a] > 0) {
      s.start[a] = lo + owned[a] - hw;
      s.count[a] = hw;
    } else if (d[a] < 0) {
      s.start[a] = lo;
      s.count[a] = hw;
    } else {
      s.start[a] = lo;
      s.count[a] = owned[a];
    }
  }
  return s;
}

/**
 * @brief Recv slab: the `hw`-thick halo ring on the `d` side.
 *
 * Inactive axes span the owned core, matching face-only MPI subarrays
 * (`create_padded_face_types_6`). Edge and corner directions shrink the
 * active axes the same way; they do not widen the inactive axes.
 */
[[nodiscard]] inline HaloSlab padded_recv_slab(const Int3 &owned, int hw,
                                               const Int3 &d) {
  validate_padded_slab_args(owned, hw, d);
  HaloSlab s{};
  for (int a = 0; a < 3; ++a) {
    if (d[a] > 0) {
      s.start[a] = hw + owned[a];
      s.count[a] = hw;
    } else if (d[a] < 0) {
      s.start[a] = 0;
      s.count[a] = hw;
    } else {
      s.start[a] = hw;
      s.count[a] = owned[a];
    }
  }
  return s;
}

} // namespace pfc::halo
