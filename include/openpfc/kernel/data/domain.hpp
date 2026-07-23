// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file domain.hpp
 * @brief `pfc::Domain` — the canonical global Cartesian simulation domain (M1).
 *
 * @details
 * `Domain` describes the **global** discretized Cartesian space: grid size,
 * uniform spacing, physical origin (coordinate of index `(0,0,0)`), and
 * **per-axis periodicity that is actually consumed**. It is the 0.2 replacement
 * for `World<CartesianTag>` and drops the vestigial coordinate-system template
 * parameter (there was ever exactly one instantiation, `CartesianTag`, and half
 * the query surface was already Cartesian-concrete — see audit §13.2).
 *
 * Roles are split cleanly:
 * - `Domain` carries the global coordinate system (spacing/origin/periodicity)
 *   and the global grid size;
 * - `Box3i` (see `box3i.hpp`) carries any *index range* — the global index box
 *   `[0, size-1]` or a per-rank subdomain box. `World`'s old conflation of the
 *   two roles (global-vs-subworld via a shifted `m_lower`) is what produced the
 *   subworld-bounds bug (audit §4.5); with `Domain + Box3i` a subdomain's
 *   physical bounds are simply `to_coords(local_box.low/high)`.
 *
 * Coordinate transforms match the (post Pre-M0) `csys`/`world` conventions
 * exactly so the M1 consumer migration is numerically identical:
 * - `to_coords(idx)  = origin + idx * spacing`
 * - `to_indices(xyz) = lround((xyz - origin) / spacing)`  (nearest-grid-point)
 *
 * This header is additive in M1.2: nothing migrates onto it yet. `World`,
 * `csys`, and `world_*` remain untouched until M1.3–M1.5.
 *
 * @see box3i.hpp for the canonical index box
 * @see world.hpp for the (to-be-deprecated) `World` it replaces
 */

#pragma once

#include <array>
#include <cmath>
#include <ostream>
#include <stdexcept>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/data/world_types.hpp>

namespace pfc {

using types::Bool3;
using types::Int3;
using types::Real3;

/**
 * @brief The global Cartesian simulation domain.
 *
 * Immutable value type (constructed once via `domain::create`, then read-only).
 * Members are public so it is an aggregate with brace initialization; prefer the
 * validated `domain::create(...)` factories for construction.
 */
struct Domain {
  Int3 size{1, 1, 1};               ///< Global grid size {nx, ny, nz} (> 0).
  Real3 spacing{1.0, 1.0, 1.0};     ///< Grid spacing per axis (> 0).
  Real3 origin{0.0, 0.0, 0.0};      ///< Physical coordinate of index (0,0,0).
  Bool3 periodic{true, true, true}; ///< Per-axis periodicity (consumed).

  friend bool operator==(const Domain &a, const Domain &b) noexcept {
    return a.size == b.size && a.spacing == b.spacing && a.origin == b.origin &&
           a.periodic == b.periodic;
  }
  friend bool operator!=(const Domain &a, const Domain &b) noexcept {
    return !(a == b);
  }
};

namespace domain {

/**
 * @brief Full-specification, type-safe factory (mirrors `world::create`).
 * @throws std::invalid_argument if any size or spacing component is non-positive.
 */
[[nodiscard]] inline Domain create(const GridSize &size,
                                   const PhysicalOrigin &origin,
                                   const GridSpacing &spacing,
                                   const Bool3 &periodic = {true, true, true}) {
  for (int d = 0; d < 3; ++d) {
    if (size.value[d] <= 0) {
      throw std::invalid_argument(
          "Domain: grid size must be positive on every axis.");
    }
    if (spacing.value[d] <= 0.0) {
      throw std::invalid_argument(
          "Domain: grid spacing must be positive on every axis.");
    }
  }
  return Domain{size.value, spacing.value, origin.value, periodic};
}

/// Convenience: unit spacing, origin at zero, fully periodic.
[[nodiscard]] inline Domain create(const Int3 &size) {
  return create(GridSize(size), PhysicalOrigin({0.0, 0.0, 0.0}),
                GridSpacing({1.0, 1.0, 1.0}));
}

/// Convenience: custom spacing, origin at zero, fully periodic.
[[nodiscard]] inline Domain with_spacing(const Int3 &size, const Real3 &spacing,
                                         const Bool3 &periodic = {true, true,
                                                                  true}) {
  return create(GridSize(size), PhysicalOrigin({0.0, 0.0, 0.0}),
                GridSpacing(spacing), periodic);
}

/**
 * @brief Create from physical bounds. Spacing depends on periodicity per axis,
 *        matching `world::from_bounds` bit-for-bit:
 *          periodic axis     → spacing = (upper - lower) / size
 *          non-periodic axis → spacing = (upper - lower) / (size - 1)
 * @throws std::invalid_argument if any size <= 0 or upper <= lower.
 */
[[nodiscard]] inline Domain from_bounds(const Int3 &size, const Real3 &lower,
                                        const Real3 &upper,
                                        const Bool3 &periodic = {true, true, true}) {
  for (int d = 0; d < 3; ++d) {
    if (size[d] <= 0) {
      throw std::invalid_argument(
          "Domain::from_bounds: grid size must be positive.");
    }
    if (upper[d] <= lower[d]) {
      throw std::invalid_argument(
          "Domain::from_bounds: upper bound must exceed lower bound.");
    }
  }
  Real3 spacing;
  for (int d = 0; d < 3; ++d) {
    spacing[d] = periodic[d] ? (upper[d] - lower[d]) / size[d]
                             : (upper[d] - lower[d]) / (size[d] - 1);
  }
  return create(GridSize(size), PhysicalOrigin(lower), GridSpacing(spacing),
                periodic);
}

// ---- Queries -------------------------------------------------------------

[[nodiscard]] inline const Int3 &get_size(const Domain &d) noexcept {
  return d.size;
}
[[nodiscard]] inline int get_size(const Domain &d, int i) { return d.size.at(i); }

[[nodiscard]] inline const Real3 &get_spacing(const Domain &d) noexcept {
  return d.spacing;
}
[[nodiscard]] inline double get_spacing(const Domain &d, int i) {
  return d.spacing.at(i);
}

[[nodiscard]] inline const Real3 &get_origin(const Domain &d) noexcept {
  return d.origin;
}

[[nodiscard]] inline const Bool3 &get_periodic(const Domain &d) noexcept {
  return d.periodic;
}
[[nodiscard]] inline bool is_periodic(const Domain &d, int i) {
  return d.periodic.at(i);
}

/// Total number of global grid points.
[[nodiscard]] inline size_t get_total_size(const Domain &d) noexcept {
  return static_cast<size_t>(d.size[0]) * static_cast<size_t>(d.size[1]) *
         static_cast<size_t>(d.size[2]);
}

/// The global index box `[0, size-1]`.
[[nodiscard]] inline Box3i index_box(const Domain &d) noexcept {
  return Box3i::from_bounds({0, 0, 0},
                            {d.size[0] - 1, d.size[1] - 1, d.size[2] - 1});
}

/// Index → physical coordinate: `origin + idx * spacing`.
[[nodiscard]] inline Real3 to_coords(const Domain &d, const Int3 &idx) noexcept {
  Real3 xyz;
  for (int i = 0; i < 3; ++i) xyz[i] = d.origin[i] + idx[i] * d.spacing[i];
  return xyz;
}

/// Physical coordinate → nearest grid index: `lround((xyz - origin)/spacing)`.
[[nodiscard]] inline Int3 to_indices(const Domain &d, const Real3 &xyz) noexcept {
  Int3 idx;
  for (int i = 0; i < 3; ++i) {
    idx[i] = static_cast<int>(std::lround((xyz[i] - d.origin[i]) / d.spacing[i]));
  }
  return idx;
}

/// Physical coordinate of the global lower corner (index (0,0,0)).
[[nodiscard]] inline Real3 get_lower_bounds(const Domain &d) noexcept {
  return to_coords(d, {0, 0, 0});
}
/// Physical coordinate of the global upper corner (index (nx-1,ny-1,nz-1)).
[[nodiscard]] inline Real3 get_upper_bounds(const Domain &d) noexcept {
  return to_coords(d, {d.size[0] - 1, d.size[1] - 1, d.size[2] - 1});
}

/// `spacing[0]*spacing[1]*spacing[2] * nx*ny*nz` (matches `world::physical_volume`).
[[nodiscard]] inline double physical_volume(const Domain &d) noexcept {
  return d.spacing[0] * d.spacing[1] * d.spacing[2] * d.size[0] * d.size[1] *
         d.size[2];
}

[[nodiscard]] inline bool is_1d(const Domain &d) noexcept {
  return (d.size[0] > 1) && (d.size[1] == 1) && (d.size[2] == 1);
}
[[nodiscard]] inline bool is_2d(const Domain &d) noexcept {
  return (d.size[0] > 1) && (d.size[1] > 1) && (d.size[2] == 1);
}
[[nodiscard]] inline bool is_3d(const Domain &d) noexcept {
  return (d.size[0] > 1) && (d.size[1] > 1) && (d.size[2] > 1);
}
/// 1/2/3 active dimensions, or 0 when fully degenerate (matches `world`).
[[nodiscard]] inline int dimensionality(const Domain &d) noexcept {
  if (is_3d(d)) return 3;
  if (is_2d(d)) return 2;
  if (is_1d(d)) return 1;
  return 0;
}

} // namespace domain

inline std::ostream &operator<<(std::ostream &os, const Domain &d) {
  os << "Domain(size={" << d.size[0] << "," << d.size[1] << "," << d.size[2]
     << "}, spacing={" << d.spacing[0] << "," << d.spacing[1] << "," << d.spacing[2]
     << "}, origin={" << d.origin[0] << "," << d.origin[1] << "," << d.origin[2]
     << "}, periodic={" << d.periodic[0] << "," << d.periodic[1] << ","
     << d.periodic[2] << "})";
  return os;
}

} // namespace pfc
