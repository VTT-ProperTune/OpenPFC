// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file dirichlet_boundary.hpp
 * @brief Named Cartesian Dirichlet faces for stage preparation.
 *
 * @details
 * Node-centred grids put an owned sample on the wall. `enforce_owned_nodes`
 * sets that sample to `g` before the halo exchange, and the ghost layer `k`
 * is the odd mirror `u(-k) = 2 g - u(k)` (high face:
 * `u(N-1+k) = 2 g - u(N-1-k)`). The mirror uses every stored halo layer, so
 * a stencil of half-width `M` needs `storage_halo() >= M`. Ghost fill runs
 * after the exchange and overwrites a non-periodic face. It does not run
 * again after an RK update: call `enforce_owned_nodes` once more if the
 * step rewrote the wall.
 *
 * Cell-face grids put the wall halfway outside the first sample. Ghosts
 * follow `u(-k) = 2 g - u(k-1)`, and owned samples are not the boundary, so
 * they are not pinned. That is the `wave2d` closure. `wave2d` still pins
 * its endpoint rows after the wave update; that pin is the wave unknown,
 * not this ghost rule.
 *
 * A corner is the X-then-Y-then-Z composition of those one-axis mirrors.
 * Two walls that meet on one sample are written in face order
 * `xm, xp, ym, yp, zm, zp`, so the later face supplies the corner value.
 *
 * Constant and time-dependent face values call `apply_reflection_ghosts` on
 * the field's memory space. CUDA and HIP therefore fill ghosts with the
 * same mirror when the reflection kernel is compiled. There is no device
 * kernel that pins owned nodes: `bind_dirichlet` accepts host fields only.
 * A coordinate callback is host-only and throws on a device field.
 * Fourier operators keep their periodic basis.
 *
 * JSON shape, one representation:
 * `{"centering": "node", "faces": [{"face": "zm", "value": 0.0}]}`.
 * Face names are `xm`, `xp`, `ym`, `yp`, `zm`, `zp`. `centering` is
 * `node` (default) or `cell`.
 */

#include <array>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/dirichlet_ghosts.hpp>
#include <openpfc/kernel/decomposition/stage_preparation.hpp>
#include <openpfc/kernel/integrator/stage_context.hpp>

namespace pfc::communication {

/// Global sides in `ReflectionWalls` order: −x +x −y +y −z +z.
enum class CartesianFace { Xm = 0, Xp, Ym, Yp, Zm, Zp };

/// Node samples lie on the wall. Cell-face samples lie half a spacing inside.
enum class DirichletCentering { Node, CellFace };

[[nodiscard]] inline constexpr int cartesian_face_axis(CartesianFace face) noexcept {
  return static_cast<int>(face) / 2;
}

[[nodiscard]] inline constexpr bool
cartesian_face_is_low(CartesianFace face) noexcept {
  return static_cast<int>(face) % 2 == 0;
}

[[nodiscard]] inline std::string_view cartesian_face_name(CartesianFace face) {
  constexpr std::array<std::string_view, 6> names{"xm", "xp", "ym",
                                                  "yp", "zm", "zp"};
  const int index = static_cast<int>(face);
  if (index < 0 || index > 5) {
    throw std::invalid_argument("CartesianFace: index out of range");
  }
  return names[static_cast<std::size_t>(index)];
}

[[nodiscard]] inline CartesianFace cartesian_face_from_name(std::string_view name) {
  for (int index = 0; index < 6; ++index) {
    const auto face = static_cast<CartesianFace>(index);
    if (cartesian_face_name(face) == name) {
      return face;
    }
  }
  throw std::invalid_argument(
      "Cartesian face must be one of xm, xp, ym, yp, zm, zp");
}

/**
 * @brief Dirichlet data on selected Cartesian faces.
 *
 * @tparam T Field value type. JSON parsing is provided for `double`.
 */
template <typename T = double> class DirichletBoundary {
public:
  DirichletCentering centering = DirichletCentering::Node;
  /// `PhysicalFaces` rejects a Dirichlet face on a periodic axis.
  ReflectionExtent extent = ReflectionExtent::PhysicalFaces;

  void set_constant(CartesianFace face, T value) {
    FaceValue &slot = faces_[index_of(face)];
    slot = FaceValue{};
    slot.enabled = true;
    slot.constant = value;
  }

  /// Scalar `g(t)` for the whole face. Evaluated when ghosts are filled.
  void set_time_dependent(CartesianFace face, std::function<T(double)> value) {
    if (!value) {
      throw std::invalid_argument("Dirichlet time callback is empty");
    }
    FaceValue &slot = faces_[index_of(face)];
    slot = FaceValue{};
    slot.enabled = true;
    slot.time_value = std::move(value);
  }

  /// `g(x, t)` on the wall. Host fields only. `x` is the wall coordinate.
  void set_coordinate_dependent(CartesianFace face,
                                std::function<T(pfc::Real3, double)> value) {
    if (!value) {
      throw std::invalid_argument("Dirichlet coordinate callback is empty");
    }
    FaceValue &slot = faces_[index_of(face)];
    slot = FaceValue{};
    slot.enabled = true;
    slot.space_time = std::move(value);
  }

  [[nodiscard]] bool empty() const noexcept {
    for (const FaceValue &face : faces_) {
      if (face.enabled) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool coordinate_dependent() const noexcept {
    for (const FaceValue &face : faces_) {
      if (face.enabled && face.space_time) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] bool enabled(CartesianFace face) const noexcept {
    return faces_[index_of(face)].enabled;
  }

  [[nodiscard]] T evaluate(CartesianFace face, const pfc::Real3 &x,
                           double time) const {
    const FaceValue &slot = faces_[index_of(face)];
    if (!slot.enabled) {
      throw std::invalid_argument("Dirichlet face is not enabled");
    }
    if (slot.space_time) {
      return slot.space_time(x, time);
    }
    if (slot.time_value) {
      return slot.time_value(time);
    }
    return slot.constant;
  }

  /**
   * @brief Set owned node-centred wall samples to `g`. Cell-face data is
   *        left unchanged. Host fields only.
   */
  template <class Space>
  void enforce_owned_nodes(pfc::data::Field<T, Space> &field, double time) const {
    static_assert(std::is_same_v<Space, pfc::HostSpace>,
                  "Dirichlet owned nodes are pinned on host fields");
    if (centering != DirichletCentering::Node || empty()) {
      return;
    }
    reject_periodic_faces(field.domain());
    const auto size = field.global_size();
    field.for_each_owned([&](int i, int j, int k) {
      const pfc::Int3 global = field.global(i, j, k);
      const pfc::Real3 x = field.coords(i, j, k);
      bool on_wall = false;
      T value{};
      for (int side = 0; side < 6; ++side) {
        const auto face = static_cast<CartesianFace>(side);
        if (!faces_[side].enabled) {
          continue;
        }
        const int axis = side / 2;
        const bool low = side % 2 == 0;
        const bool here = low ? global[axis] == 0 : global[axis] == size[axis] - 1;
        if (!here) {
          continue;
        }
        on_wall = true;
        value = evaluate(face, x, time);
      }
      if (on_wall) {
        field(i, j, k) = value;
      }
    });
  }

  /**
   * @brief Fill Dirichlet ghosts. Does not write owned samples.
   *
   * Constant and time-dependent values use `apply_reflection_ghosts`.
   * A coordinate callback uses the same odd recurrence on the host, with
   * `g` taken at the wall point of each mirrored axis.
   */
  template <class Space>
  void apply_ghosts(pfc::data::Field<T, Space> &field, double time) const {
    if (empty()) {
      return;
    }
    reject_periodic_faces(field.domain());
    if (coordinate_dependent()) {
      if constexpr (!std::is_same_v<Space, pfc::HostSpace>) {
        throw std::invalid_argument(
            "Dirichlet coordinate callbacks run on host fields only");
      } else {
        apply_coordinate_host(field, time);
      }
      return;
    }
    apply_reflection_ghosts(field, scalar_walls(time), extent,
                            /*mirror_disabled_corners=*/true);
  }

  /**
   * @brief Register owned pinning and ghost fill for one host field.
   *
   * The boundary, the field and @p time must outlive later `prepare` calls.
   * Drive `prepare` with `dirichlet_stage_requirements` so the pin is
   * exchanged before ghosts are filled. `prepare` is pre-evaluation: pin
   * again after a step that rewrites owned walls.
   */
  template <class Space>
  void bind_dirichlet(StagePreparationService<T> &prep, std::string_view name,
                      pfc::data::Field<T, Space> &field, const double &time) const {
    static_assert(std::is_same_v<Space, pfc::HostSpace>,
                  "bind_dirichlet pins owned nodes on host fields");
    prep.set_owned_update(
        name, [this, &field, &time] { enforce_owned_nodes(field, time); });
    prep.set_ghost_update(name,
                          [this, &field, &time] { apply_ghosts(field, time); });
  }

private:
  struct FaceValue {
    bool enabled = false;
    T constant{};
    std::function<T(double)> time_value;
    std::function<T(pfc::Real3, double)> space_time;
  };

  std::array<FaceValue, 6> faces_{};

  static int index_of(CartesianFace face) {
    const int index = static_cast<int>(face);
    if (index < 0 || index > 5) {
      throw std::invalid_argument("CartesianFace: index out of range");
    }
    return index;
  }

  void reject_periodic_faces(const pfc::Domain &domain) const {
    if (extent == ReflectionExtent::AllOutside) {
      return;
    }
    for (int side = 0; side < 6; ++side) {
      if (!faces_[side].enabled) {
        continue;
      }
      const int axis = side / 2;
      if (pfc::domain::is_periodic(domain, axis)) {
        throw std::invalid_argument(
            std::string("Dirichlet face '") +
            std::string(cartesian_face_name(static_cast<CartesianFace>(side))) +
            "' is on a periodic axis");
      }
    }
  }

  [[nodiscard]] ReflectionWalls<T> scalar_walls(double time) const {
    ReflectionWalls<T> walls;
    ReflectionSide<T> *const sides[6] = {&walls.xm, &walls.xp, &walls.ym,
                                         &walls.yp, &walls.zm, &walls.zp};
    const BoundaryCentering center = centering == DirichletCentering::Node
                                         ? BoundaryCentering::Node
                                         : BoundaryCentering::CellFace;
    for (int side = 0; side < 6; ++side) {
      if (!faces_[side].enabled) {
        continue;
      }
      const auto face = static_cast<CartesianFace>(side);
      sides[side]->enabled = true;
      sides[side]->parity = ReflectionParity::Odd;
      sides[side]->centering = center;
      sides[side]->value = evaluate(face, pfc::Real3{}, time);
    }
    return walls;
  }

  void apply_coordinate_host(pfc::data::Field<T, pfc::HostSpace> &field,
                             double time) const {
    ReflectionWalls<T> walls;
    ReflectionSide<T> *const sides[6] = {&walls.xm, &walls.xp, &walls.ym,
                                         &walls.yp, &walls.zm, &walls.zp};
    const BoundaryCentering center_tag = centering == DirichletCentering::Node
                                             ? BoundaryCentering::Node
                                             : BoundaryCentering::CellFace;
    for (int side = 0; side < 6; ++side) {
      if (!faces_[side].enabled) {
        continue;
      }
      sides[side]->enabled = true;
      sides[side]->parity = ReflectionParity::Odd;
      sides[side]->centering = center_tag;
    }
    detail::ReflectionGeometry<T> geom =
        detail::build_reflection_geometry(field, walls, extent);
    geom.skip_disabled = 1;
    detail::validate_reflection_geometry(geom);
    const detail::ReflectionVisitPlan plan = detail::plan_reflection_visits(geom);
    const pfc::Real3 origin = field.origin();
    const pfc::Real3 spacing = field.spacing();
    const DirichletCentering center = centering;
    for (int slab_index = 0; slab_index < plan.n; ++slab_index) {
      const detail::ReflectionSlab &slab = plan.slabs[slab_index];
      for (int k = slab.begin[2]; k < slab.end[2]; ++k) {
        for (int j = slab.begin[1]; j < slab.end[1]; ++j) {
          for (int i = slab.begin[0]; i < slab.end[0]; ++i) {
            const int local[3] = {i, j, k};
            int src[3] = {i, j, k};
            T scale = T{1};
            T offset = T{0};
            int any = 0;
            int cancel = 0;
            for (int axis = 0; axis < 3; ++axis) {
              const int global = geom.low[axis] + local[axis];
              const int applied =
                  detail::reflection_side_applies(geom, axis, global);
              if (applied < 0) {
                if (geom.skip_disabled == 0) {
                  cancel = 1;
                }
                continue;
              }
              if (applied == 0) {
                continue;
              }
              const int side = axis * 2 + (global < 0 ? 0 : 1);
              const int mirrored = detail::mirrored_global(global, geom.nglob[axis],
                                                           geom.cell_face[side]);
              src[axis] = mirrored - geom.low[axis];
              any = 1;
              pfc::Real3 x = field.coords(i, j, k);
              if (center == DirichletCentering::Node) {
                const int wall = global < 0 ? 0 : geom.nglob[axis] - 1;
                x[axis] = origin[axis] + static_cast<double>(wall) * spacing[axis];
              } else {
                const double wall =
                    global < 0 ? -0.5 : static_cast<double>(geom.nglob[axis]) - 0.5;
                x[axis] = origin[axis] + wall * spacing[axis];
              }
              const T wall_value =
                  evaluate(static_cast<CartesianFace>(side), x, time);
              offset = static_cast<T>(2) * wall_value - offset;
              scale = static_cast<T>(-scale);
            }
            if (any && cancel == 0) {
              field(i, j, k) = scale * field(src[0], src[1], src[2]) + offset;
            }
          }
        }
      }
    }
  }
};

/**
 * @brief Stage requirements for a bound Dirichlet operator.
 *
 * Boundary update is on, and owned pins run before the halo exchange.
 * Ghost fill is the registered ghost update, which runs after that exchange.
 */
[[nodiscard]] inline StagePreparationRequirements
dirichlet_stage_requirements(const pfc::integrator::StageContext &ctx) {
  StagePreparationRequirements req = pfc::integrator::requirements_from(ctx, true);
  req.ordering = BoundaryHaloOrder::BoundaryThenHalo;
  return req;
}

} // namespace pfc::communication
