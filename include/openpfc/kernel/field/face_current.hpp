// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file face_current.hpp
 * @brief Conservative divergence of a model-supplied Cartesian face current.
 *
 * @details
 * `divergence_separated` evaluates only the constitutive law
 * `F = a_face (p_high - p_low) / h`. This operator is the remainder:
 * telescoping `+div G` of caller-supplied face values, with a wall policy
 * and an outer-flux budget. Constitutive laws stay in the caller.
 *
 * Sign convention, matching `divergence_separated`:
 *
 *     R_c = Σ_axes (G_{c+½e} − G_{c−½e}) / h_axis,     ∂t u = R + sources
 *
 * `G_f` is the +e component of the bracket in `∂t u = ∇·G`. A model that
 * stores a physical current `J` with `∂t u = −∇·J` passes `G = −J`.
 *
 * Face arrays are x-fastest within each plane:
 *
 *     x-faces: (nx+1)*ny*nz, index i + j*(nx+1) + k*(nx+1)*ny, i in [0, nx]
 *     y-faces: nx*(ny+1)*nz, index i + j*nx     + k*nx*(ny+1), j in [0, ny]
 *     z-faces: nx*ny*(nz+1), index i + j*nx     + k*nx*ny,     k in [0, nz]
 *
 * An inactive axis (global extent 1) is skipped; its span may be empty.
 * The operator does not exchange halos. Shared faces between ranks, and
 * the periodic seam, are ordinary faces that both sides fill from their
 * own data. Global telescoping holds iff those independent evaluations
 * are bitwise equal.
 *
 * @see face_flux.hpp
 */

#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>

#include <openpfc/kernel/data/host_device.hpp>

namespace pfc::field::fd {

/// How a global non-periodic side is closed.
///
/// This is the face-current law. It is not a scalar ghost fill.
/// `pfc::communication::ReflectionWalls` mirrors padded samples; an even
/// mirror does not make a general anisotropic `G` zero. A model that wants
/// `G_f = 0` selects `Zero` here.
enum class WallPolicy {
  Model,     ///< leave the caller-supplied face value (functor or buffer)
  Zero,      ///< impose G_f = 0; the functor is not used on this side
  Prescribed ///< impose G_f = Wall::value (the +e component)
};

struct Wall {
  WallPolicy policy = WallPolicy::Zero;
  double value = 0.0; ///< used only when policy is Prescribed
};

/// Six sides of the owned box, in the order −x, +x, −y, +y, −z, +z.
struct Walls {
  Wall xm{}, xp{}, ym{}, yp{}, zm{}, zp{};
};

[[nodiscard]] inline Walls closed_walls() noexcept { return {}; }

[[nodiscard]] inline Walls model_walls() noexcept {
  Walls w;
  w.xm.policy = WallPolicy::Model;
  w.xp.policy = WallPolicy::Model;
  w.ym.policy = WallPolicy::Model;
  w.yp.policy = WallPolicy::Model;
  w.zm.policy = WallPolicy::Model;
  w.zp.policy = WallPolicy::Model;
  return w;
}

[[nodiscard]] inline std::size_t n_cells(int nx, int ny, int nz) noexcept {
  return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
         static_cast<std::size_t>(nz);
}
[[nodiscard]] inline std::size_t n_xfaces(int nx, int ny, int nz) noexcept {
  return static_cast<std::size_t>(nx + 1) * static_cast<std::size_t>(ny) *
         static_cast<std::size_t>(nz);
}
[[nodiscard]] inline std::size_t n_yfaces(int nx, int ny, int nz) noexcept {
  return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny + 1) *
         static_cast<std::size_t>(nz);
}
[[nodiscard]] inline std::size_t n_zfaces(int nx, int ny, int nz) noexcept {
  return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
         static_cast<std::size_t>(nz + 1);
}

[[nodiscard]] inline std::size_t xface_index(int nx, int ny, int i, int j,
                                             int k) noexcept {
  return static_cast<std::size_t>(i) +
         static_cast<std::size_t>(j) * static_cast<std::size_t>(nx + 1) +
         static_cast<std::size_t>(k) * static_cast<std::size_t>(nx + 1) *
             static_cast<std::size_t>(ny);
}
[[nodiscard]] inline std::size_t yface_index(int nx, int ny, int i, int j,
                                             int k) noexcept {
  return static_cast<std::size_t>(i) +
         static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
         static_cast<std::size_t>(k) * static_cast<std::size_t>(nx) *
             static_cast<std::size_t>(ny + 1);
}
[[nodiscard]] inline std::size_t zface_index(int nx, int ny, int i, int j,
                                             int k) noexcept {
  return static_cast<std::size_t>(i) +
         static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
         static_cast<std::size_t>(k) * static_cast<std::size_t>(nx) *
             static_cast<std::size_t>(ny);
}
[[nodiscard]] inline std::size_t cell_index(int nx, int ny, int i, int j,
                                            int k) noexcept {
  return static_cast<std::size_t>(i) +
         static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
         static_cast<std::size_t>(k) * static_cast<std::size_t>(nx) *
             static_cast<std::size_t>(ny);
}

inline void check_face_extents(int nx, int ny, int nz) {
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    throw std::invalid_argument("pfc::field::fd: nx, ny and nz must be positive");
  }
}

/**
 * Pointwise `+div G`, with the same face-current convention as `fill_faces`.
 *
 * The current functor receives `(axis, i, j, k)` for the low-index cell of
 * a face. It must evaluate shared faces identically on both ranks. For a
 * device evaluator, it must be host/device callable and contain device-safe
 * data. This object owns no fields, buffers, halo exchange or launch policy.
 * Its coordinates are local owned cells, including local extents of one.
 *
 * Construct with the `face_divergence` factory, which validates extents
 * and active-axis spacing on the host. Walls act on owned-box outer faces,
 * exactly as for `fill_faces`; select Model on MPI cuts and periodic seams.
 * A caller captures its padded device samples in the current functor,
 * refreshes their halos in stage preparation, then embeds this evaluator
 * in the existing resident RK composite evaluator:
 *
 * @code
 * auto div = face_divergence(current, nx, ny, nz, {true, true, true},
 *                            dx, dy, dz, model_walls());
 * // Inside a host/device per-cell RHS evaluator:
 * double dc = div(i, j, k);
 * @endcode
 *
 * Values outside the owned box must be supplied by the current's halo
 * samples. No host fallback or automatic halo exchange is performed.
 */
template <class Flux> struct FaceDivergence {
  Flux flux;
  int extent[3];
  bool active[3];
  double inverse_spacing[3];
  Wall walls[6]; // -x,+x,-y,+y,-z,+z

  OPENPFC_HD double operator()(int i, int j, int k) const {
    double div = 0.0;
    const int cell[3] = {i, j, k};
    for (int axis = 0; axis < 3; ++axis) {
      if (!active[axis]) continue;
      const Wall &lo = walls[2 * axis];
      const Wall &hi = walls[2 * axis + 1];
      const double low =
          cell[axis] == 0 && lo.policy != WallPolicy::Model
              ? (lo.policy == WallPolicy::Zero ? 0.0 : lo.value)
              : flux(axis, i - (axis == 0), j - (axis == 1), k - (axis == 2));
      const double high =
          cell[axis] == extent[axis] - 1 && hi.policy != WallPolicy::Model
              ? (hi.policy == WallPolicy::Zero ? 0.0 : hi.value)
              : flux(axis, i, j, k);
      div += (high - low) * inverse_spacing[axis];
    }
    return div;
  }
};

/// Create a per-cell evaluator for host or resident RK evaluation.
template <class Flux>
[[nodiscard]] FaceDivergence<Flux>
face_divergence(Flux flux, int nx, int ny, int nz, std::array<bool, 3> active,
                double dx, double dy, double dz, const Walls &walls) {
  check_face_extents(nx, ny, nz);
  const double spacing[3] = {dx, dy, dz};
  double inverse[3]{};
  for (int axis = 0; axis < 3; ++axis) {
    if (active[axis]) {
      if (!(spacing[axis] > 0.0))
        throw std::invalid_argument(
            "pfc::field::fd::face_divergence: active spacing");
      inverse[axis] = 1.0 / spacing[axis];
    }
  }
  return {std::move(flux),
          {nx, ny, nz},
          {active[0], active[1], active[2]},
          {inverse[0], inverse[1], inverse[2]},
          {walls.xm, walls.xp, walls.ym, walls.yp, walls.zm, walls.zp}};
}

/// Overwrite @p rhs with `+div G` of the three face arrays.
inline void face_divergence(std::span<const double> gx, std::span<const double> gy,
                            std::span<const double> gz, std::span<double> rhs,
                            int nx, int ny, int nz, std::array<bool, 3> active,
                            double dx, double dy, double dz) {
  check_face_extents(nx, ny, nz);
  const std::size_t n = n_cells(nx, ny, nz);
  if (rhs.size() != n) {
    throw std::invalid_argument(
        "pfc::field::fd::face_divergence: rhs must have length nx*ny*nz");
  }
  const bool use_x = active[0];
  const bool use_y = active[1];
  const bool use_z = active[2];
  if (use_x && gx.size() != n_xfaces(nx, ny, nz)) {
    throw std::invalid_argument("pfc::field::fd::face_divergence: gx length");
  }
  if (use_y && gy.size() != n_yfaces(nx, ny, nz)) {
    throw std::invalid_argument("pfc::field::fd::face_divergence: gy length");
  }
  if (use_z && gz.size() != n_zfaces(nx, ny, nz)) {
    throw std::invalid_argument("pfc::field::fd::face_divergence: gz length");
  }
  if (use_x && !(dx > 0.0))
    throw std::invalid_argument("pfc::field::fd::face_divergence: dx");
  if (use_y && !(dy > 0.0))
    throw std::invalid_argument("pfc::field::fd::face_divergence: dy");
  if (use_z && !(dz > 0.0))
    throw std::invalid_argument("pfc::field::fd::face_divergence: dz");
  const double invx = use_x ? 1.0 / dx : 0.0;
  const double invy = use_y ? 1.0 / dy : 0.0;
  const double invz = use_z ? 1.0 / dz : 0.0;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        double div = 0.0;
        if (use_x) {
          div += (gx[xface_index(nx, ny, i + 1, j, k)] -
                  gx[xface_index(nx, ny, i, j, k)]) *
                 invx;
        }
        if (use_y) {
          div += (gy[yface_index(nx, ny, i, j + 1, k)] -
                  gy[yface_index(nx, ny, i, j, k)]) *
                 invy;
        }
        if (use_z) {
          div += (gz[zface_index(nx, ny, i, j, k + 1)] -
                  gz[zface_index(nx, ny, i, j, k)]) *
                 invz;
        }
        rhs[cell_index(nx, ny, i, j, k)] = div;
      }
    }
  }
}

/// Surface integral of `G · n_out` over the owned-box boundary.
/// `Σ_cells R * dx dy dz` equals this value to round-off.
[[nodiscard]] inline double outer_flux(std::span<const double> gx,
                                       std::span<const double> gy,
                                       std::span<const double> gz, int nx, int ny,
                                       int nz, std::array<bool, 3> active, double dx,
                                       double dy, double dz) {
  check_face_extents(nx, ny, nz);
  const bool use_x = active[0];
  const bool use_y = active[1];
  const bool use_z = active[2];
  double s = 0.0;
  if (use_x) {
    const double area = dy * dz;
    for (int k = 0; k < nz; ++k)
      for (int j = 0; j < ny; ++j) {
        s += gx[xface_index(nx, ny, nx, j, k)] * area;
        s -= gx[xface_index(nx, ny, 0, j, k)] * area;
      }
  }
  if (use_y) {
    const double area = dx * dz;
    for (int k = 0; k < nz; ++k)
      for (int i = 0; i < nx; ++i) {
        s += gy[yface_index(nx, ny, i, ny, k)] * area;
        s -= gy[yface_index(nx, ny, i, 0, k)] * area;
      }
  }
  if (use_z) {
    const double area = dx * dy;
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        s += gz[zface_index(nx, ny, i, j, nz)] * area;
        s -= gz[zface_index(nx, ny, i, j, 0)] * area;
      }
  }
  return s;
}

/// Apply Zero / Prescribed to the outer faces of @p gx, @p gy, @p gz.
/// Model sides are left unchanged.
inline void apply_walls(std::span<double> gx, std::span<double> gy,
                        std::span<double> gz, int nx, int ny, int nz,
                        std::array<bool, 3> active, const Walls &walls) {
  check_face_extents(nx, ny, nz);
  auto set_side = [](std::span<double> faces, std::size_t idx, const Wall &w) {
    if (w.policy == WallPolicy::Zero)
      faces[idx] = 0.0;
    else if (w.policy == WallPolicy::Prescribed)
      faces[idx] = w.value;
  };
  if (active[0]) {
    for (int k = 0; k < nz; ++k)
      for (int j = 0; j < ny; ++j) {
        set_side(gx, xface_index(nx, ny, 0, j, k), walls.xm);
        set_side(gx, xface_index(nx, ny, nx, j, k), walls.xp);
      }
  }
  if (active[1]) {
    for (int k = 0; k < nz; ++k)
      for (int i = 0; i < nx; ++i) {
        set_side(gy, yface_index(nx, ny, i, 0, k), walls.ym);
        set_side(gy, yface_index(nx, ny, i, ny, k), walls.yp);
      }
  }
  if (active[2]) {
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        set_side(gz, zface_index(nx, ny, i, j, 0), walls.zm);
        set_side(gz, zface_index(nx, ny, i, j, nz), walls.zp);
      }
  }
}

/**
 * Fill every face from @p flux, then apply @p walls.
 *
 * `flux(axis, i, j, k)` returns the +e component of G on the face whose
 * lower-index cell is `(i, j, k)`. `axis` is 0, 1 or 2. The lower index
 * along that axis is in `[-1, n)` (the −e wall is `−1`). Zero and
 * Prescribed walls skip the functor.
 *
 * Inactive axes are not filled.
 */
template <class Flux>
void fill_faces(Flux &&flux, std::span<double> gx, std::span<double> gy,
                std::span<double> gz, int nx, int ny, int nz,
                std::array<bool, 3> active, const Walls &walls) {
  check_face_extents(nx, ny, nz);
  auto skip = [](int face, int n, const Wall &lo, const Wall &hi) -> const Wall * {
    if (face == 0 && lo.policy != WallPolicy::Model) return &lo;
    if (face == n && hi.policy != WallPolicy::Model) return &hi;
    return nullptr;
  };
  if (active[0]) {
    if (gx.size() != n_xfaces(nx, ny, nz))
      throw std::invalid_argument("pfc::field::fd::fill_faces: gx length");
    for (int k = 0; k < nz; ++k)
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i <= nx; ++i) {
          const Wall *w = skip(i, nx, walls.xm, walls.xp);
          const std::size_t idx = xface_index(nx, ny, i, j, k);
          gx[idx] = w ? (w->policy == WallPolicy::Zero ? 0.0 : w->value)
                      : flux(0, i - 1, j, k);
        }
  }
  if (active[1]) {
    if (gy.size() != n_yfaces(nx, ny, nz))
      throw std::invalid_argument("pfc::field::fd::fill_faces: gy length");
    for (int k = 0; k < nz; ++k)
      for (int j = 0; j <= ny; ++j)
        for (int i = 0; i < nx; ++i) {
          const Wall *w = skip(j, ny, walls.ym, walls.yp);
          const std::size_t idx = yface_index(nx, ny, i, j, k);
          gy[idx] = w ? (w->policy == WallPolicy::Zero ? 0.0 : w->value)
                      : flux(1, i, j - 1, k);
        }
  }
  if (active[2]) {
    if (gz.size() != n_zfaces(nx, ny, nz))
      throw std::invalid_argument("pfc::field::fd::fill_faces: gz length");
    for (int k = 0; k <= nz; ++k)
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
          const Wall *w = skip(k, nz, walls.zm, walls.zp);
          const std::size_t idx = zface_index(nx, ny, i, j, k);
          gz[idx] = w ? (w->policy == WallPolicy::Zero ? 0.0 : w->value)
                      : flux(2, i, j, k - 1);
        }
  }
}

} // namespace pfc::field::fd
