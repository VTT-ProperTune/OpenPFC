// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file robin_boundary.hpp
 * @brief Named Cartesian Robin faces for stage preparation.
 *
 * @details
 * Node-centred faces implement `alpha u + beta du/dn = gamma`. The normal
 * `n` points out of the domain: `-1` on a low face and `+1` on a high face.
 * `beta = 0` is node Dirichlet (`u = gamma / alpha`), reusing the odd
 * mirror and the owned-node pin. `alpha = 0` is Neumann; a zero `gamma`
 * is the even mirror. Cell-face Robin is not represented.
 *
 * For a nonzero `beta` the ghost layer `k` continues a linear odd part so
 * that every central first derivative at the wall equals the slope implied
 * by the relation:
 * - low face: `u(-k) = u(k) - 2 k h du/dx`, with
 *   `du/dx = (alpha u_wall - gamma) / beta`;
 * - high face: `u(N-1+k) = u(N-1-k) + 2 k h du/dx`, with
 *   `du/dx = (gamma - alpha u_wall) / beta`.
 * That identity is exact for every tabulated central first derivative,
 * because those stencils are exact on linears and the even part cancels.
 * It reproduces quadratic polynomials. A cubic ghost is not reproduced.
 * Fill all `storage_halo()` layers; a stencil of half-width `M` needs
 * `storage_halo() >= M`.
 *
 * Corners use the reflection visit plan. Each outside axis applies its
 * map in order X, then Y, then Z. A Robin jump is added from the interior
 * mirror's wall sample. A later Dirichlet axis negates an earlier jump.
 * The Laplacian reads face ghosts, not the diagonal corner.
 *
 * `coefficient_tolerance` defaults to 0, so only an exact `(alpha, beta)
 * = (0, 0)` pair is singular. A positive tolerance is absolute: the pair
 * is rejected when both `|alpha|` and `|beta|` are at most the tolerance,
 * and again at fill time when both `|alpha|` and `|beta| / h` are. A small
 * nonzero `beta` with a larger `alpha` stays Robin; it is not switched to
 * Dirichlet.
 *
 * Constant coefficients and a time-dependent `gamma` are host-only. There
 * is no device Robin kernel. `bind_robin` pins Dirichlet faces before the
 * halo exchange and fills ghosts after it. `prepare` does not repeat the
 * pin after an RK update.
 *
 * JSON shape, one representation:
 * `{"faces": [{"face": "xm", "alpha": 1.0, "beta": 0.5, "gamma": -0.25}]}`.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <openpfc/kernel/decomposition/dirichlet_boundary.hpp>

namespace pfc::communication {

/**
 * @brief Robin data on selected Cartesian faces.
 *
 * @tparam T Field value type. JSON parsing is provided for `double`.
 */
template <typename T = double> class RobinBoundary {
public:
  /// Absolute tolerance for `(alpha, beta)`. Zero rejects only `(0, 0)`.
  T coefficient_tolerance = T{0};
  /// `PhysicalFaces` rejects a Robin face on a periodic axis.
  ReflectionExtent extent = ReflectionExtent::PhysicalFaces;

  void set_constant(CartesianFace face, T alpha, T beta, T gamma) {
    check_pair(alpha, beta);
    if (!std::isfinite(static_cast<double>(gamma))) {
      throw std::invalid_argument("Robin gamma must be finite");
    }
    FaceValue &slot = faces_[index_of(face)];
    slot = FaceValue{};
    slot.enabled = true;
    slot.alpha = alpha;
    slot.beta = beta;
    slot.gamma = gamma;
  }

  /// `gamma(t)` for the whole face. `alpha` and `beta` stay constant.
  void set_time_dependent_gamma(CartesianFace face, T alpha, T beta,
                                std::function<T(double)> gamma) {
    check_pair(alpha, beta);
    if (!gamma) {
      throw std::invalid_argument("Robin gamma callback is empty");
    }
    FaceValue &slot = faces_[index_of(face)];
    slot = FaceValue{};
    slot.enabled = true;
    slot.alpha = alpha;
    slot.beta = beta;
    slot.gamma_time = std::move(gamma);
  }

  [[nodiscard]] bool empty() const noexcept {
    for (const FaceValue &face : faces_) {
      if (face.enabled) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool enabled(CartesianFace face) const noexcept {
    return faces_[index_of(face)].enabled;
  }

  [[nodiscard]] T alpha(CartesianFace face) const { return slot(face).alpha; }

  [[nodiscard]] T beta(CartesianFace face) const { return slot(face).beta; }

  [[nodiscard]] T gamma(CartesianFace face, double time) const {
    return evaluate_gamma(slot(face), time);
  }

  /**
   * @brief Pin owned samples on faces with `beta = 0` to `gamma / alpha`.
   *        Other faces are left unchanged. Host fields only.
   */
  template <class Space>
  void enforce_dirichlet_nodes(pfc::data::Field<T, Space> &field,
                               double time) const {
    static_assert(std::is_same_v<Space, pfc::HostSpace>,
                  "Robin Dirichlet nodes are pinned on host fields");
    if (empty()) {
      return;
    }
    reject_periodic_faces(field.domain());
    check_scaled_faces(field.spacing());
    const auto size = field.global_size();
    field.for_each_owned([&](int i, int j, int k) {
      const pfc::Int3 global = field.global(i, j, k);
      bool on_wall = false;
      T value{};
      for (int side = 0; side < 6; ++side) {
        const FaceValue &face = faces_[side];
        if (!face.enabled || face.beta != T{0}) {
          continue;
        }
        const int axis = side / 2;
        const bool low = side % 2 == 0;
        const bool here = low ? global[axis] == 0 : global[axis] == size[axis] - 1;
        if (!here) {
          continue;
        }
        on_wall = true;
        value = evaluate_gamma(face, time) / face.alpha;
      }
      if (on_wall) {
        field(i, j, k) = value;
      }
    });
  }

  /**
   * @brief Fill Robin ghosts. Does not write owned samples.
   *
   * Dirichlet faces (`beta = 0`) use the odd mirror. Other faces add the
   * linear jump documented above. Host fields only.
   */
  template <class Space>
  void apply_ghosts(pfc::data::Field<T, Space> &field, double time) const {
    static_assert(std::is_same_v<Space, pfc::HostSpace>,
                  "Robin ghosts are filled on host fields");
    if (empty() || field.storage_halo() < 1) {
      return;
    }
    reject_periodic_faces(field.domain());
    check_scaled_faces(field.spacing());
    apply_host(field, time);
  }

  /**
   * @brief Register the Dirichlet pin and the ghost fill for one host field.
   *
   * The boundary, the field and @p time must outlive later `prepare` calls.
   * Drive `prepare` with `robin_stage_requirements`.
   */
  template <class Space>
  void bind_robin(StagePreparationService<T> &prep, std::string_view name,
                  pfc::data::Field<T, Space> &field, const double &time) const {
    static_assert(std::is_same_v<Space, pfc::HostSpace>,
                  "bind_robin fills ghosts on host fields");
    prep.set_owned_update(
        name, [this, &field, &time] { enforce_dirichlet_nodes(field, time); });
    prep.set_ghost_update(name,
                          [this, &field, &time] { apply_ghosts(field, time); });
  }

private:
  struct FaceValue {
    bool enabled = false;
    T alpha{};
    T beta{};
    T gamma{};
    std::function<T(double)> gamma_time;
  };

  std::array<FaceValue, 6> faces_{};

  static int index_of(CartesianFace face) {
    const int index = static_cast<int>(face);
    if (index < 0 || index > 5) {
      throw std::invalid_argument("CartesianFace: index out of range");
    }
    return index;
  }

  [[nodiscard]] const FaceValue &slot(CartesianFace face) const {
    const FaceValue &face_value = faces_[index_of(face)];
    if (!face_value.enabled) {
      throw std::invalid_argument("Robin face is not enabled");
    }
    return face_value;
  }

  [[nodiscard]] T tolerance() const {
    const double stored = static_cast<double>(coefficient_tolerance);
    if (!(stored > 0.0) || !std::isfinite(stored)) {
      return T{0};
    }
    return coefficient_tolerance;
  }

  void check_pair(T alpha, T beta) const {
    if (!std::isfinite(static_cast<double>(alpha)) ||
        !std::isfinite(static_cast<double>(beta))) {
      throw std::invalid_argument("Robin alpha and beta must be finite");
    }
    const double tol = static_cast<double>(tolerance());
    if (std::hypot(static_cast<double>(alpha), static_cast<double>(beta)) <= tol) {
      throw std::invalid_argument(
          "Robin coefficients are singular: alpha and beta are both within "
          "tolerance");
    }
  }

  void check_scaled_faces(const pfc::Real3 &spacing) const {
    const double tol = static_cast<double>(tolerance());
    for (int side = 0; side < 6; ++side) {
      const FaceValue &face = faces_[side];
      if (!face.enabled) {
        continue;
      }
      const double h = spacing[side / 2];
      if (!(h > 0.0) || !std::isfinite(h)) {
        throw std::invalid_argument("Robin spacing must be positive");
      }
      const double scaled = std::max(std::abs(static_cast<double>(face.alpha)),
                                     std::abs(static_cast<double>(face.beta)) / h);
      if (scaled <= tol) {
        throw std::invalid_argument(
            std::string("Robin face '") +
            std::string(cartesian_face_name(static_cast<CartesianFace>(side))) +
            "' is singular at this spacing");
      }
    }
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
            std::string("Robin face '") +
            std::string(cartesian_face_name(static_cast<CartesianFace>(side))) +
            "' is on a periodic axis");
      }
    }
  }

  [[nodiscard]] static T evaluate_gamma(const FaceValue &face, double time) {
    const T value = face.gamma_time ? face.gamma_time(time) : face.gamma;
    if (!std::isfinite(static_cast<double>(value))) {
      throw std::invalid_argument("Robin gamma must be finite");
    }
    return value;
  }

  void apply_host(pfc::data::Field<T, pfc::HostSpace> &field, double time) const {
    ReflectionWalls<T> walls;
    ReflectionSide<T> *const sides[6] = {&walls.xm, &walls.xp, &walls.ym,
                                         &walls.yp, &walls.zm, &walls.zp};
    for (int side = 0; side < 6; ++side) {
      if (!faces_[side].enabled) {
        continue;
      }
      sides[side]->enabled = true;
      sides[side]->parity =
          faces_[side].beta == T{0} ? ReflectionParity::Odd : ReflectionParity::Even;
      sides[side]->centering = BoundaryCentering::Node;
    }
    detail::ReflectionGeometry<T> geom =
        detail::build_reflection_geometry(field, walls, extent);
    geom.skip_disabled = 1;
    detail::validate_reflection_geometry(geom);
    const detail::ReflectionVisitPlan plan = detail::plan_reflection_visits(geom);
    const pfc::Real3 spacing = field.spacing();
    for (int slab_index = 0; slab_index < plan.n; ++slab_index) {
      const detail::ReflectionSlab &slab = plan.slabs[slab_index];
      for (int k = slab.begin[2]; k < slab.end[2]; ++k) {
        for (int j = slab.begin[1]; j < slab.end[1]; ++j) {
          for (int i = slab.begin[0]; i < slab.end[0]; ++i) {
            write_ghost(field, geom, i, j, k, spacing, time);
          }
        }
      }
    }
  }

  void write_ghost(pfc::data::Field<T, pfc::HostSpace> &field,
                   const detail::ReflectionGeometry<T> &geom, int i, int j, int k,
                   const pfc::Real3 &spacing, double time) const {
    const int local[3] = {i, j, k};
    int src[3] = {i, j, k};
    int layer[3] = {0, 0, 0};
    int side_of[3] = {-1, -1, -1};
    int cancel = 0;
    int any = 0;
    for (int axis = 0; axis < 3; ++axis) {
      const int global = geom.low[axis] + local[axis];
      const int applied = detail::reflection_side_applies(geom, axis, global);
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
      const int mirrored =
          detail::mirrored_global(global, geom.nglob[axis], geom.cell_face[side]);
      src[axis] = mirrored - geom.low[axis];
      layer[axis] = global < 0 ? -global : global - (geom.nglob[axis] - 1);
      side_of[axis] = side;
      any = 1;
    }
    if (!any || cancel != 0) {
      return;
    }
    T scale = T{1};
    T offset = T{0};
    for (int axis = 0; axis < 3; ++axis) {
      if (side_of[axis] < 0) {
        continue;
      }
      const int side = side_of[axis];
      const FaceValue &face = faces_[side];
      const T wall_gamma = evaluate_gamma(face, time);
      if (face.beta == T{0}) {
        const T dirichlet = wall_gamma / face.alpha;
        offset = static_cast<T>(2) * dirichlet - offset;
        scale = static_cast<T>(-scale);
        continue;
      }
      int wall[3] = {src[0], src[1], src[2]};
      const int global = geom.low[axis] + local[axis];
      const int wall_global = global < 0 ? 0 : geom.nglob[axis] - 1;
      wall[axis] = wall_global - geom.low[axis];
      const T u_wall = field(wall[0], wall[1], wall[2]);
      const T partial = global < 0 ? (face.alpha * u_wall - wall_gamma) / face.beta
                                   : (wall_gamma - face.alpha * u_wall) / face.beta;
      if (!std::isfinite(static_cast<double>(partial))) {
        throw std::invalid_argument("Robin slope is not finite");
      }
      const T sign = global < 0 ? T{-1} : T{1};
      offset += sign * static_cast<T>(2 * layer[axis]) *
                static_cast<T>(spacing[axis]) * partial;
    }
    const T mirrored = field(src[0], src[1], src[2]);
    field(i, j, k) = scale * mirrored + offset;
  }
};

/**
 * @brief Stage requirements for a bound Robin operator.
 *
 * Dirichlet pins run before the halo exchange. Ghost fill runs after it.
 */
[[nodiscard]] inline StagePreparationRequirements
robin_stage_requirements(const pfc::integrator::StageContext &ctx) {
  StagePreparationRequirements req = pfc::integrator::requirements_from(ctx, true);
  req.ordering = BoundaryHaloOrder::BoundaryThenHalo;
  return req;
}

} // namespace pfc::communication
