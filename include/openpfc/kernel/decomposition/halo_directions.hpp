// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file halo_directions.hpp
 * @brief Customizable halo direction sets for OpenPFC exchangers.
 *
 * @details
 * Halo exchangers in OpenPFC historically hard-code their direction list:
 * `HaloExchanger`, `PaddedHaloExchanger`, `PersistentHaloExchanger`,
 * `PaddedDeviceHaloExchanger`, and `BatchedPaddedDeviceHalo` all loop over
 * the **6 axis-aligned face directions** (`±X`, `±Y`, `±Z`);
 * `FullPaddedDeviceHalo` runs **3 widening passes** that touch all 26
 * neighbour cells. Helpers exist (`pfc::halo::Connectivity{Faces,Edges,All}`,
 * `decomposition::find_face_neighbors`, `find_all_neighbors`) but no
 * exchanger lets you say "exchange only `±X`" or "this is a 2D slab,
 * skip `±Z`."
 *
 * `HaloDirectionSet` is the **first-class abstraction** for that decision: it
 * stores a deduplicated, validated list of `Int3` unit vectors (each component
 * in `{-1, 0, 1}`, never the zero vector) and exposes named presets for the
 * common cases:
 *
 *   | Preset      | Size | Members                                |
 *   |-------------|------|----------------------------------------|
 *   | `Axes2D`    |   4  | `±X, ±Y`                               |
 *   | `Full2D`    |   8  | axes + 4 XY corners                    |
 *   | `Axes3D`    |   6  | `±X, ±Y, ±Z`                           |
 *   | `Full3D`    |  26  | axes + edges + corners (3D stencil)    |
 *
 * Custom direction lists are constructed via `HaloDirectionSet(dirs)` and
 * validated at construction time. A `HaloDirectionSelector` callable lets
 * callers override the active set on a per-rank basis (e.g. shrink the set
 * near non-periodic boundaries) without rewriting every ctor.
 *
 * The helper `direction_to_face_slot(d)` maps an axis-aligned direction to the
 * canonical 6-face slot order `(+X, -X, +Y, -Y, +Z, -Z)` shared by
 * `pfc::halo::create_face_types_6` and `create_padded_face_types_6`. Non-axis
 * directions return `-1`.
 *
 * For back-compat with `pfc::halo::Connectivity`, `from_connectivity()`
 * translates the legacy enum to a preset based on the requested dimensionality.
 *
 * @see decomposition_neighbors.hpp for the canonical 26-direction enumeration
 *      reused by `Full3D()`.
 * @see halo_exchange.hpp / padded_halo_exchange.hpp / halo_persistent.hpp
 *      for the CPU exchangers that consume this type.
 * @see sparse_halo_exchange.hpp + halo_face_layout.hpp for the sparse
 *      `pfc::SparseHaloExchanger` and the `make_structured_halos` helper
 *      that turns a `HaloDirectionSet` into a `RemoteHalo` list.
 * @see runtime/cuda/padded_device_halo_exchange.hpp /
 *      runtime/cuda/full_padded_device_halo.hpp for the CUDA exchangers.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/decomposition/halo_pattern.hpp>

namespace pfc::halo {

/**
 * @brief Set of unit-vector neighbour directions used by a halo exchanger.
 *
 * Each entry is a `pfc::types::Int3` whose components are in `{-1, 0, 1}` and
 * which is **not** the zero vector. The set is **deduplicated** at
 * construction. The container preserves insertion order so callers can
 * predict iteration order, but membership is what matters for the exchangers.
 */
struct HaloDirectionSet {
  using Int3 = pfc::types::Int3;

  std::vector<Int3> dirs;

  HaloDirectionSet() = default;

  /**
   * @brief Construct from an explicit list of direction vectors.
   *
   * Validates each entry: components must be in `{-1, 0, 1}` and not all zero.
   * Duplicate entries are silently dropped (kept once, in first-seen order).
   *
   * @throws std::invalid_argument on a `{0,0,0}` entry or an out-of-range
   *         component.
   */
  explicit HaloDirectionSet(std::vector<Int3> directions) {
    dirs.reserve(directions.size());
    for (const auto &d : directions) {
      validate_direction_(d);
      if (!contains(d)) {
        dirs.push_back(d);
      }
    }
  }

  /// Number of directions in the set (`0..26`).
  [[nodiscard]] std::size_t size() const noexcept { return dirs.size(); }

  /// `true` when no direction is configured.
  [[nodiscard]] bool empty() const noexcept { return dirs.empty(); }

  /// Membership test on the active direction list.
  [[nodiscard]] bool contains(const Int3 &d) const noexcept {
    return std::find(dirs.begin(), dirs.end(), d) != dirs.end();
  }

  /// Equality compares the dir lists element-wise (order-sensitive).
  friend bool operator==(const HaloDirectionSet &a,
                         const HaloDirectionSet &b) noexcept {
    return a.dirs == b.dirs;
  }
  friend bool operator!=(const HaloDirectionSet &a,
                         const HaloDirectionSet &b) noexcept {
    return !(a == b);
  }

private:
  static void validate_direction_(const Int3 &d) {
    for (int c : d) {
      if (c < -1 || c > 1) {
        throw std::invalid_argument(
            "HaloDirectionSet: direction component must be in {-1,0,1}, got " +
            std::to_string(c));
      }
    }
    if (d[0] == 0 && d[1] == 0 && d[2] == 0) {
      throw std::invalid_argument(
          "HaloDirectionSet: zero direction {0,0,0} is not a valid neighbour");
    }
  }
};

/**
 * @brief Per-rank override callback for the active direction set.
 *
 * If supplied to an exchanger constructor, the exchanger calls
 * `selector(rank)` for its own rank and uses that result instead of the
 * uniform fallback `HaloDirectionSet`. Lets callers shrink the set near
 * boundaries without changing every ctor signature.
 */
using HaloDirectionSelector = std::function<HaloDirectionSet(int rank)>;

namespace presets {

/**
 * @brief 4-direction in-plane axes for 2D slab problems (`±X`, `±Y`).
 *
 * Use for fields whose `nz == 1` and whose stencil never reads `k±1` (e.g.
 * the Kobayashi 2D phase-field driver). Skipping `±Z` removes the `nx*ny`
 * face slabs that turn into `MPI`-to-self transfers when the process grid
 * has extent 1 along Z.
 */
[[nodiscard]] inline HaloDirectionSet Axes2D() {
  using Dir = HaloDirectionSet::Int3;
  return HaloDirectionSet(
      std::vector<Dir>{Dir{1, 0, 0}, Dir{-1, 0, 0}, Dir{0, 1, 0}, Dir{0, -1, 0}});
}

/**
 * @brief 8-direction in-plane stencil neighbours (axes + 4 XY corners).
 *
 * For 2D stencils that read diagonal neighbours `(i±1, j±1)`, e.g. mixed
 * `u_xy` derivatives or 9-point Laplacians.
 */
[[nodiscard]] inline HaloDirectionSet Full2D() {
  using Dir = HaloDirectionSet::Int3;
  return HaloDirectionSet(
      std::vector<Dir>{Dir{1, 0, 0}, Dir{-1, 0, 0}, Dir{0, 1, 0}, Dir{0, -1, 0},
                       Dir{1, 1, 0}, Dir{1, -1, 0}, Dir{-1, 1, 0}, Dir{-1, -1, 0}});
}

/**
 * @brief 6-direction axis-aligned 3D stencil (`±X, ±Y, ±Z`).
 *
 * The historical default for every face exchanger; matches the 7-point
 * Laplacian and other axis-aligned stencils.
 */
[[nodiscard]] inline HaloDirectionSet Axes3D() {
  using Dir = HaloDirectionSet::Int3;
  return HaloDirectionSet(std::vector<Dir>{Dir{1, 0, 0}, Dir{-1, 0, 0}, Dir{0, 1, 0},
                                           Dir{0, -1, 0}, Dir{0, 0, 1},
                                           Dir{0, 0, -1}});
}

/**
 * @brief 26-direction full 3D stencil (faces + edges + corners).
 *
 * Mirrors the enumeration in `decomposition::find_all_neighbors` (single
 * source of truth) and matches the default for `FullPaddedDeviceHalo`.
 */
[[nodiscard]] inline HaloDirectionSet Full3D() {
  using Dir = HaloDirectionSet::Int3;
  std::vector<Dir> v;
  v.reserve(26);
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        if (dx == 0 && dy == 0 && dz == 0) {
          continue;
        }
        v.push_back(Dir{dx, dy, dz});
      }
    }
  }
  return HaloDirectionSet(std::move(v));
}

} // namespace presets

/**
 * @brief Map an axis-aligned direction to the 6-face slot index.
 *
 * Returns the canonical slot order shared by `create_face_types_6` and
 * `create_padded_face_types_6`:
 *
 *   `+X = 0, -X = 1, +Y = 2, -Y = 3, +Z = 4, -Z = 5`
 *
 * Returns `-1` for any non-face direction (edges and corners).
 */
[[nodiscard]] inline int
direction_to_face_slot(const HaloDirectionSet::Int3 &d) noexcept {
  static constexpr std::array<HaloDirectionSet::Int3, 6> kFaceDirs = {
      {HaloDirectionSet::Int3{1, 0, 0}, HaloDirectionSet::Int3{-1, 0, 0},
       HaloDirectionSet::Int3{0, 1, 0}, HaloDirectionSet::Int3{0, -1, 0},
       HaloDirectionSet::Int3{0, 0, 1}, HaloDirectionSet::Int3{0, 0, -1}}};
  for (int i = 0; i < 6; ++i) {
    if (kFaceDirs[static_cast<std::size_t>(i)] == d) {
      return i;
    }
  }
  return -1;
}

/**
 * @brief Deterministic, peer-independent tag offset for any 26-direction vector.
 *
 * @details
 * Used by `pfc::halo::make_structured_halos` (and any other helper that needs
 * to put unique, symmetric MPI tags on a `RemoteHalo` exchange):
 *
 *   - For the **6 axis-aligned faces** (`±X, ±Y, ±Z`) the result equals
 *     `direction_to_face_slot(d)` — i.e. the canonical `0..5` slot order
 *     shared with `create_face_types_6` (and the templated brick FD
 *     Laplacians `field::fd::laplacian_periodic_separated<Order>`).
 *   - For the **20 non-axis directions** (12 edges + 8 corners) the result is
 *     `6 + (dx + 1) + 3 * (dy + 1) + 9 * (dz + 1)`, i.e. a deterministic
 *     `6..32` encoding that depends only on the direction vector.
 *
 * Because the encoding depends only on `d` (not on rank, not on local dir
 * order), two ranks negotiating an exchange agree on tags as long as one
 * rank uses `direction_to_canonical_tag(d)` for its *send* side and
 * `direction_to_canonical_tag(-d)` for its *recv* side; the peer rank does
 * the symmetric thing with `-d`, so `peer.send_tag == self.recv_tag` and
 * vice versa.
 *
 * @return Non-negative tag offset in `[0, 33)`.
 */
[[nodiscard]] inline int
direction_to_canonical_tag(const HaloDirectionSet::Int3 &d) noexcept {
  const int slot = direction_to_face_slot(d);
  if (slot >= 0) {
    return slot;
  }
  return 6 + (d[0] + 1) + 3 * (d[1] + 1) + 9 * (d[2] + 1);
}

/**
 * @brief Inverse of `direction_to_face_slot`: 6-face slot to direction vector.
 *
 * @throws std::out_of_range when `slot` is not in `[0, 6)`.
 */
[[nodiscard]] inline HaloDirectionSet::Int3 face_slot_to_direction(int slot) {
  static constexpr std::array<HaloDirectionSet::Int3, 6> kFaceDirs = {
      {HaloDirectionSet::Int3{1, 0, 0}, HaloDirectionSet::Int3{-1, 0, 0},
       HaloDirectionSet::Int3{0, 1, 0}, HaloDirectionSet::Int3{0, -1, 0},
       HaloDirectionSet::Int3{0, 0, 1}, HaloDirectionSet::Int3{0, 0, -1}}};
  if (slot < 0 || slot >= 6) {
    throw std::out_of_range("face_slot_to_direction: slot must be in [0,6), got " +
                            std::to_string(slot));
  }
  return kFaceDirs[static_cast<std::size_t>(slot)];
}

/**
 * @brief Translate the legacy `pfc::halo::Connectivity` enum to a direction set.
 *
 * @param c     Connectivity pattern.
 * @param ndims Dimensionality hint (`2` or `3`). For `Faces` returns
 *              `Axes2D()`/`Axes3D()`; for `Edges` returns `Full2D()` (8-dir)
 *              when `ndims == 2` and `Full3D()` otherwise; for `All` always
 *              returns `Full3D()`.
 *
 * @throws std::invalid_argument if `ndims` is not `2` or `3`.
 */
[[nodiscard]] inline HaloDirectionSet from_connectivity(Connectivity c, int ndims) {
  if (ndims != 2 && ndims != 3) {
    throw std::invalid_argument("from_connectivity: ndims must be 2 or 3, got " +
                                std::to_string(ndims));
  }
  switch (c) {
  case Connectivity::Faces:
    return (ndims == 2) ? presets::Axes2D() : presets::Axes3D();
  case Connectivity::Edges:
    return (ndims == 2) ? presets::Full2D() : presets::Full3D();
  case Connectivity::All: return presets::Full3D();
  }
  return presets::Axes3D();
}

/**
 * @brief Resolve the active direction set for a given rank.
 *
 * If `selector` is non-empty, the result is `selector(rank)`; otherwise it
 * is `fallback`. Helper used uniformly by every exchanger ctor that accepts
 * a `HaloDirectionSelector`.
 */
[[nodiscard]] inline HaloDirectionSet
resolve_direction_set(const HaloDirectionSet &fallback,
                      const HaloDirectionSelector &selector, int rank) {
  if (selector) {
    return selector(rank);
  }
  return fallback;
}

} // namespace pfc::halo
