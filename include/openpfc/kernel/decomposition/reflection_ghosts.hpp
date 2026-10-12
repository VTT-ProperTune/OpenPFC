// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file reflection_ghosts.hpp
 * @brief Even and odd reflection of padded ghosts on global faces.
 *
 * @details
 * Sibling of `HaloExchange`, not a second boundary framework. The exchange
 * remains transport. `StagePreparationService::set_reflection` runs this fill
 * after that exchange when a stage asks for a boundary update. Device fields
 * use the same `ReflectionWalls` from `runtime/gpu/reflection_ghosts_gpu.hpp`.
 *
 * Two placements:
 * - `BoundaryCentering::Node` — samples sit on the physical bound (global
 *   indices `0` and `N-1`). The mirror of `-k` is `+k`, and the mirror of
 *   `N-1+k` is `N-1-k`. This is the placement `apply_dirichlet_ghosts` uses.
 * - `BoundaryCentering::CellFace` — the wall lies on the face outside the
 *   first and last samples. The mirror of `-k` is `k-1`, and the mirror of
 *   `N-1+k` is `N-k`.
 *
 * `ReflectionParity::Even` copies the mirror. `Odd` stores
 * `2 * value - mirror`. A corner or edge that is outside on several axes is
 * the composition of those one-axis maps in order X, then Y, then Z — the
 * same order as a full halo's widening passes. The source is the in-domain
 * (or already exchanged) sample, so the corner does not depend on an
 * intermediate ghost.
 *
 * `ReflectionExtent::PhysicalFaces` (the default) writes only global
 * non-periodic faces this rank touches. Periodic halos stay with the
 * exchange. `AllOutside` also overwrites a periodic wrap; the one-axis
 * Dirichlet helper uses that.
 *
 * A ghost layer is valid when its mirror lands in the global index range
 * and in this rank's padded box. Node centering needs `halo_width < N` on
 * that axis. Cell-face centering needs `halo_width <= N`.
 *
 * Even reflection makes a centered normal difference of this scalar zero
 * when the centering matches the stencil. It does not set a face current
 * to zero. Anisotropic constitutive fluxes stay a model choice and use
 * `pfc::field::fd::WallPolicy` in `face_current.hpp`.
 *
 * @see stage_preparation.hpp
 * @see dirichlet_ghosts.hpp
 * @see face_current.hpp
 */

#include <stdexcept>
#include <string>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/host_device.hpp>

namespace pfc::communication {

/// Copy the mirror, or reflect it through a wall value.
enum class ReflectionParity { Even, Odd };

/// Where the physical bound sits relative to the samples.
enum class BoundaryCentering { Node, CellFace };

/// Which outside indices the fill may overwrite.
enum class ReflectionExtent {
  /// Non-periodic global faces touched by this rank.
  PhysicalFaces,
  /// Every local index outside `[0, N)`, including a periodic wrap.
  AllOutside
};

/// One global side. `value` is used only for odd reflection.
template <typename T = double> struct ReflectionSide {
  bool enabled = false;
  ReflectionParity parity = ReflectionParity::Even;
  BoundaryCentering centering = BoundaryCentering::Node;
  T value{};
};

/// Six sides, in the same order as `pfc::field::fd::Walls`: −x +x −y +y −z +z.
template <typename T = double> struct ReflectionWalls {
  ReflectionSide<T> xm{}, xp{}, ym{}, yp{}, zm{}, zp{};
};

/// Enable every side with the same parity, centering, and odd-wall value.
template <typename T = double>
[[nodiscard]] ReflectionWalls<T>
reflection_walls(ReflectionParity parity,
                 BoundaryCentering centering = BoundaryCentering::Node,
                 T value = T{}) {
  ReflectionWalls<T> walls;
  const ReflectionSide<T> side{true, parity, centering, value};
  walls.xm = side;
  walls.xp = side;
  walls.ym = side;
  walls.yp = side;
  walls.zm = side;
  walls.zp = side;
  return walls;
}

namespace detail {

template <typename T> struct ReflectionGeometry {
  int low[3]{};
  int nloc[3]{};
  int nglob[3]{};
  int halo = 0;
  int periodic[3]{};
  int all_outside = 0;
  int enabled[6]{};
  int odd[6]{};
  int cell_face[6]{};
  T value[6]{};
};

template <typename T> struct ReflectionSample {
  int si = 0;
  int sj = 0;
  int sk = 0;
  T scale{};
  T offset{};
  int write = 0;
};

/// 0: leave this axis, 1: mirror it, -1: outside a physical side that is off.
template <typename T>
OPENPFC_INLINE_HD int reflection_side_applies(const ReflectionGeometry<T> &geom,
                                              int axis, int g) noexcept {
  const int n = geom.nglob[axis];
  if (g >= 0 && g < n) {
    return 0;
  }
  const int low_side = g < 0 ? 1 : 0;
  if (low_side) {
    if (geom.low[axis] != 0) {
      return 0;
    }
  } else if (geom.low[axis] + geom.nloc[axis] != n) {
    return 0;
  }
  const int physical = geom.all_outside || !geom.periodic[axis];
  if (!physical) {
    return 0;
  }
  const int side = axis * 2 + (low_side ? 0 : 1);
  if (!geom.enabled[side]) {
    return -1;
  }
  return 1;
}

OPENPFC_INLINE_HD int mirrored_global(int g, int n, int cell_face) noexcept {
  if (g < 0) {
    return cell_face ? (-1 - g) : (-g);
  }
  return cell_face ? (2 * n - 1 - g) : (2 * (n - 1) - g);
}

template <typename T>
OPENPFC_INLINE_HD ReflectionSample<T>
reflection_sample(const ReflectionGeometry<T> &geom, int i, int j, int k) noexcept {
  ReflectionSample<T> out;
  out.scale = T{1};
  out.offset = T{0};
  const int local[3] = {i, j, k};
  int src[3] = {i, j, k};
  int cancel = 0;
  int any = 0;
  for (int axis = 0; axis < 3; ++axis) {
    const int g = geom.low[axis] + local[axis];
    const int applied = reflection_side_applies(geom, axis, g);
    if (applied < 0) {
      cancel = 1;
      continue;
    }
    if (applied == 0) {
      continue;
    }
    const int side = axis * 2 + (g < 0 ? 0 : 1);
    const int gm = mirrored_global(g, geom.nglob[axis], geom.cell_face[side]);
    src[axis] = gm - geom.low[axis];
    any = 1;
    if (geom.odd[side]) {
      out.offset = static_cast<T>(2) * geom.value[side] - out.offset;
      out.scale = static_cast<T>(-out.scale);
    }
  }
  out.si = src[0];
  out.sj = src[1];
  out.sk = src[2];
  out.write = (any && !cancel) ? 1 : 0;
  return out;
}

template <typename FieldT>
[[nodiscard]] ReflectionGeometry<typename FieldT::value_type>
build_reflection_geometry(const FieldT &u,
                          const ReflectionWalls<typename FieldT::value_type> &walls,
                          ReflectionExtent extent) {
  using T = typename FieldT::value_type;
  ReflectionGeometry<T> geom;
  const auto low = u.box().low;
  const auto nloc = u.local_size();
  const auto nglob = u.global_size();
  geom.halo = u.storage_halo();
  geom.all_outside = extent == ReflectionExtent::AllOutside ? 1 : 0;
  const ReflectionSide<T> sides[6] = {walls.xm, walls.xp, walls.ym,
                                      walls.yp, walls.zm, walls.zp};
  for (int axis = 0; axis < 3; ++axis) {
    geom.low[axis] = low[axis];
    geom.nloc[axis] = nloc[axis];
    geom.nglob[axis] = nglob[axis];
    geom.periodic[axis] = pfc::domain::is_periodic(u.domain(), axis) ? 1 : 0;
  }
  for (int s = 0; s < 6; ++s) {
    geom.enabled[s] = sides[s].enabled ? 1 : 0;
    geom.odd[s] = sides[s].parity == ReflectionParity::Odd ? 1 : 0;
    geom.cell_face[s] = sides[s].centering == BoundaryCentering::CellFace ? 1 : 0;
    geom.value[s] = sides[s].value;
  }
  return geom;
}

template <typename T>
void validate_reflection_geometry(const ReflectionGeometry<T> &geom) {
  if (geom.halo < 1) {
    return;
  }
  for (int axis = 0; axis < 3; ++axis) {
    for (int low_side = 1; low_side >= 0; --low_side) {
      const int probe = low_side ? -1 : geom.nglob[axis];
      if (reflection_side_applies(geom, axis, probe) != 1) {
        continue;
      }
      const int side = axis * 2 + (low_side ? 0 : 1);
      for (int layer = 1; layer <= geom.halo; ++layer) {
        const int g = low_side ? -layer : geom.nglob[axis] + layer - 1;
        const int gm = mirrored_global(g, geom.nglob[axis], geom.cell_face[side]);
        if (gm < 0 || gm >= geom.nglob[axis]) {
          throw std::out_of_range(
              "apply_reflection_ghosts: mirror falls outside the global "
              "index range (axis=" +
              std::to_string(axis) + ", global=" + std::to_string(g) + ", mirror=" +
              std::to_string(gm) + ", N=" + std::to_string(geom.nglob[axis]) + ")");
        }
        const int src = gm - geom.low[axis];
        if (src < -geom.halo || src >= geom.nloc[axis] + geom.halo) {
          throw std::out_of_range(
              "apply_reflection_ghosts: mirrored index out of padded range "
              "(axis=" +
              std::to_string(axis) + ", global=" + std::to_string(g) +
              ", mirror=" + std::to_string(gm) + ")");
        }
      }
    }
  }
}

template <typename T>
void apply_reflection_host(pfc::data::Field<T, pfc::HostSpace> &u,
                           const ReflectionGeometry<T> &geom) {
  const int hw = geom.halo;
  const int i1 = geom.nloc[0] + hw;
  const int j1 = geom.nloc[1] + hw;
  const int k1 = geom.nloc[2] + hw;
  for (int k = -hw; k < k1; ++k) {
    for (int j = -hw; j < j1; ++j) {
      for (int i = -hw; i < i1; ++i) {
        const ReflectionSample<T> sample = reflection_sample(geom, i, j, k);
        if (!sample.write) {
          continue;
        }
        u(i, j, k) =
            sample.scale * u(sample.si, sample.sj, sample.sk) + sample.offset;
      }
    }
  }
}

} // namespace detail

/**
 * @brief Reflect padded ghosts on the sides enabled in @p walls.
 *
 * Owned samples are not written. A side that is outside the domain but
 * disabled leaves that ghost unchanged, including corners that touch it.
 * Call after the halo exchange when a tangential halo is part of the source
 * (periodic neighbour, or a widened edge).
 *
 * @throws std::out_of_range if an enabled ghost layer does not mirror into
 *         the global index range and this rank's padded box.
 */
template <typename T>
void apply_reflection_ghosts(
    pfc::data::Field<T, pfc::HostSpace> &u, const ReflectionWalls<T> &walls,
    ReflectionExtent extent = ReflectionExtent::PhysicalFaces) {
  if (u.storage_halo() < 1) {
    return;
  }
  const detail::ReflectionGeometry<T> geom =
      detail::build_reflection_geometry(u, walls, extent);
  detail::validate_reflection_geometry(geom);
  detail::apply_reflection_host(u, geom);
}

} // namespace pfc::communication
