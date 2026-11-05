// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file level_set.hpp
 * @brief First-order periodic Hamilton–Jacobi level set on a Cartesian grid.
 *
 * The contract is `docs/design/level_set.md`. The caller owns one periodic
 * cell-centred field. The stencil reaches one neighbour, so the halo width
 * is 1. An x-slab reads that neighbour from the caller. Coefficients are
 * code units. This header does not allocate a signed-distance field per
 * grain and it does not move a grain boundary.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace pfc::field::level_set {

/// One neighbour. The upwind stencil does not read farther.
inline constexpr int level_set_halo_width = 1;

enum class LevelSetMode { Advection, NormalSpeed, Redistance };

/// Cell counts and spacing. Cell `i` sits at `(i + 1/2) hx`.
struct LevelSetGrid {
  int nx{0};
  int ny{0};
  int nz{0};
  double hx{0.0};
  double hy{0.0};
  double hz{0.0};
};

/// Volume of the negative set and the step that produced the field.
struct LevelSetStep {
  double negative_volume{0.0};
  double dt{0.0};
};

/// Volume-weighted centre of cells where `phi < 0`.
struct LevelSetCentroid {
  double x{0.0};
  double y{0.0};
  double z{0.0};
  double volume{0.0};
};

[[nodiscard]] inline std::size_t level_set_count(const LevelSetGrid &grid) {
  if (grid.nx < 1 || grid.ny < 1 || grid.nz < 1) return 0;
  return static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.ny) *
         static_cast<std::size_t>(grid.nz);
}

/// Scalar characteristic-speed estimate for a single active direction.
/// For multi-direction advection use level_set_advection_stable_dt instead.
[[nodiscard]] inline double level_set_stable_dt(const LevelSetGrid &grid,
                                                double max_speed, double cfl) {
  if (!(grid.hx > 0.0) || !(grid.hy > 0.0) || !(grid.hz > 0.0))
    throw std::invalid_argument("level-set spacing must be positive");
  if (!(max_speed > 0.0) || !std::isfinite(max_speed))
    throw std::invalid_argument("level-set speed must be positive");
  if (!(cfl > 0.0) || cfl > 1.0 || !std::isfinite(cfl))
    throw std::invalid_argument("level-set CFL must lie in (0, 1]");
  const double h = std::min(grid.hx, std::min(grid.hy, grid.hz));
  return cfl * h / max_speed;
}

/// Monotone SSP-RK2/upwind CFL for a spatially varying velocity field.
/// Sum directional Courant contributions at each cell, then take the maximum:
/// dt <= cfl / max(|vx|/hx + |vy|/hy + |vz|/hz).
/// A singleton axis contributes no derivative. Zero velocity has no finite
/// advection CFL bound; this helper rejects it rather than returning infinity.
[[nodiscard]] inline double
level_set_advection_stable_dt(const LevelSetGrid &grid, std::span<const double> vx,
                              std::span<const double> vy,
                              std::span<const double> vz, double cfl) {
  const std::size_t n = level_set_count(grid);
  if (grid.nx < 2 || grid.ny < 1 || grid.nz < 1)
    throw std::invalid_argument("level-set advection needs valid dimensions");
  if (!(grid.hx > 0.0) || !(grid.hy > 0.0) || !(grid.hz > 0.0) ||
      !std::isfinite(grid.hx) || !std::isfinite(grid.hy) ||
      !std::isfinite(grid.hz))
    throw std::invalid_argument("level-set advection spacing invalid");
  if (n == 0 || vx.size() != n || vy.size() != n || vz.size() != n)
    throw std::invalid_argument("level-set advection velocity sizes invalid");
  if (!(cfl > 0.0) || cfl > 1.0 || !std::isfinite(cfl))
    throw std::invalid_argument("level-set CFL must lie in (0, 1]");
  double max_rate = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(vx[i]) || !std::isfinite(vy[i]) || !std::isfinite(vz[i]))
      throw std::invalid_argument("level-set advection speed is not finite");
    const double rate = std::abs(vx[i]) / grid.hx +
                        (grid.ny > 1 ? std::abs(vy[i]) / grid.hy : 0.0) +
                        (grid.nz > 1 ? std::abs(vz[i]) / grid.hz : 0.0);
    if (!std::isfinite(rate))
      throw std::invalid_argument("level-set advection Courant rate overflow");
    max_rate = std::max(max_rate, rate);
  }
  if (!(max_rate > 0.0))
    throw std::invalid_argument("level-set advection has zero characteristic speed");
  return cfl / max_rate;
}

[[nodiscard]] inline LevelSetCentroid
level_set_negative_centroid(const LevelSetGrid &grid, std::span<const double> phi) {
  const std::size_t n = level_set_count(grid);
  if (n == 0 || phi.size() != n)
    throw std::invalid_argument("level-set centroid size does not match the grid");
  double volume = 0.0;
  double sx = 0.0;
  double sy = 0.0;
  double sz = 0.0;
  const double cell = grid.hx * grid.hy * grid.hz;
  for (int k = 0; k < grid.nz; ++k) {
    for (int j = 0; j < grid.ny; ++j) {
      for (int i = 0; i < grid.nx; ++i) {
        const std::size_t p =
            static_cast<std::size_t>(i + grid.nx * (j + grid.ny * k));
        if (!(phi[p] < 0.0)) continue;
        volume += cell;
        sx += cell * (static_cast<double>(i) + 0.5) * grid.hx;
        sy += cell * (static_cast<double>(j) + 0.5) * grid.hy;
        sz += cell * (static_cast<double>(k) + 0.5) * grid.hz;
      }
    }
  }
  LevelSetCentroid out;
  out.volume = volume;
  if (volume > 0.0) {
    out.x = sx / volume;
    out.y = sy / volume;
    out.z = sz / volume;
  }
  return out;
}

/// First outward zero of `phi` along +x, measured from cell centres.
[[nodiscard]] inline double level_set_zero_crossing_x(const LevelSetGrid &grid,
                                                      std::span<const double> phi,
                                                      int j, int k, int i_start) {
  const std::size_t n = level_set_count(grid);
  if (n == 0 || phi.size() != n)
    throw std::invalid_argument("level-set crossing size does not match the grid");
  if (j < 0 || k < 0 || j >= grid.ny || k >= grid.nz || i_start < 0 ||
      i_start >= grid.nx)
    throw std::invalid_argument("level-set crossing index is outside the grid");
  for (int i = i_start; i < grid.nx - 1; ++i) {
    const std::size_t p = static_cast<std::size_t>(i + grid.nx * (j + grid.ny * k));
    const std::size_t q = p + 1;
    const double a = phi[p];
    const double b = phi[q];
    if (a == 0.0) return (static_cast<double>(i) + 0.5) * grid.hx;
    if ((a < 0.0 && b >= 0.0) || (a > 0.0 && b <= 0.0)) {
      const double t = a / (a - b);
      return (static_cast<double>(i) + 0.5 + t) * grid.hx;
    }
  }
  throw std::invalid_argument("level-set crossing found no sign change along +x");
}

namespace detail {

inline int wrap_index(int i, int n) {
  const int m = i % n;
  return m < 0 ? m + n : m;
}

inline void validate_grid(const LevelSetGrid &grid) {
  if (grid.nx < 2 || grid.ny < 1 || grid.nz < 1)
    throw std::invalid_argument("level-set grid needs at least two cells in x");
  if (!(grid.hx > 0.0) || !(grid.hy > 0.0) || !(grid.hz > 0.0) ||
      !std::isfinite(grid.hx) || !std::isfinite(grid.hy) || !std::isfinite(grid.hz))
    throw std::invalid_argument("level-set spacing must be positive");
}

inline void require_field(std::span<const double> values, std::size_t n,
                          const char *what, bool needed) {
  if (!needed) return;
  if (values.size() != n)
    throw std::invalid_argument(std::string("level-set ") + what +
                                " size does not match the grid");
  for (double v : values) {
    if (!std::isfinite(v))
      throw std::domain_error(std::string("nonfinite level-set ") + what);
  }
}

inline double godunov_square(double speed, double backward, double forward) {
  if (speed >= 0.0) {
    const double a = backward > 0.0 ? backward : 0.0;
    const double b = forward < 0.0 ? forward : 0.0;
    return a * a > b * b ? a * a : b * b;
  }
  const double a = backward < 0.0 ? backward : 0.0;
  const double b = forward > 0.0 ? forward : 0.0;
  return a * a > b * b ? a * a : b * b;
}

inline double sample_x(std::span<const double> owned, std::span<const double> xm,
                       std::span<const double> xp, int i, int j, int k, int nx,
                       int ny, int nz) {
  j = wrap_index(j, ny);
  k = wrap_index(k, nz);
  const std::size_t plane = static_cast<std::size_t>(j + ny * k);
  if (i == -1) return xm[plane];
  if (i == nx) return xp[plane];
  const std::size_t p = static_cast<std::size_t>(i + nx * (j + ny * k));
  return owned[p];
}

inline void rate_at(const LevelSetGrid &grid, std::span<const double> phi,
                    std::span<const double> xm, std::span<const double> xp,
                    LevelSetMode mode, std::span<const double> vx,
                    std::span<const double> vy, std::span<const double> vz,
                    std::span<const double> vn, std::span<const double> phi0, int i,
                    int j, int k, double &out) {
  const double c = sample_x(phi, xm, xp, i, j, k, grid.nx, grid.ny, grid.nz);
  const double xl = sample_x(phi, xm, xp, i - 1, j, k, grid.nx, grid.ny, grid.nz);
  const double xr = sample_x(phi, xm, xp, i + 1, j, k, grid.nx, grid.ny, grid.nz);
  const double yl = sample_x(phi, xm, xp, i, j - 1, k, grid.nx, grid.ny, grid.nz);
  const double yr = sample_x(phi, xm, xp, i, j + 1, k, grid.nx, grid.ny, grid.nz);
  const double zl = sample_x(phi, xm, xp, i, j, k - 1, grid.nx, grid.ny, grid.nz);
  const double zr = sample_x(phi, xm, xp, i, j, k + 1, grid.nx, grid.ny, grid.nz);
  const double dmx = (c - xl) / grid.hx;
  const double dpx = (xr - c) / grid.hx;
  const double dmy = grid.ny == 1 ? 0.0 : (c - yl) / grid.hy;
  const double dpy = grid.ny == 1 ? 0.0 : (yr - c) / grid.hy;
  const double dmz = grid.nz == 1 ? 0.0 : (c - zl) / grid.hz;
  const double dpz = grid.nz == 1 ? 0.0 : (zr - c) / grid.hz;
  if (mode == LevelSetMode::Advection) {
    const std::size_t p = static_cast<std::size_t>(i + grid.nx * (j + grid.ny * k));
    const double ux = vx[p] > 0.0 ? vx[p] * dmx : (vx[p] < 0.0 ? vx[p] * dpx : 0.0);
    const double uy = vy[p] > 0.0 ? vy[p] * dmy : (vy[p] < 0.0 ? vy[p] * dpy : 0.0);
    const double uz = vz[p] > 0.0 ? vz[p] * dmz : (vz[p] < 0.0 ? vz[p] * dpz : 0.0);
    out = -(ux + uy + uz);
    return;
  }
  double speed = 0.0;
  if (mode == LevelSetMode::NormalSpeed) {
    const std::size_t p = static_cast<std::size_t>(i + grid.nx * (j + grid.ny * k));
    speed = vn[p];
  } else {
    const std::size_t p = static_cast<std::size_t>(i + grid.nx * (j + grid.ny * k));
    const double h = std::min(grid.hx, std::min(grid.hy, grid.hz));
    const double base = phi0[p];
    speed = base / std::sqrt(base * base + h * h);
  }
  const double mag =
      std::sqrt(godunov_square(speed, dmx, dpx) + godunov_square(speed, dmy, dpy) +
                godunov_square(speed, dmz, dpz));
  out = mode == LevelSetMode::Redistance ? -speed * mag + speed : -speed * mag;
}

inline void fill_periodic_x_halos(const LevelSetGrid &grid,
                                  std::span<const double> phi, std::span<double> xm,
                                  std::span<double> xp) {
  for (int k = 0; k < grid.nz; ++k) {
    for (int j = 0; j < grid.ny; ++j) {
      const std::size_t s = static_cast<std::size_t>(j + grid.ny * k);
      xm[s] =
          phi[static_cast<std::size_t>((grid.nx - 1) + grid.nx * (j + grid.ny * k))];
      xp[s] = phi[static_cast<std::size_t>(grid.nx * (j + grid.ny * k))];
    }
  }
}

} // namespace detail

/// Owned-cell rate. `xm` and `xp` are the x-halos, each of length `ny * nz`.
inline void level_set_rate_x_slab(
    const LevelSetGrid &grid, std::span<const double> phi,
    std::span<const double> halo_xm, std::span<const double> halo_xp,
    LevelSetMode mode, std::span<const double> velocity_x,
    std::span<const double> velocity_y, std::span<const double> velocity_z,
    std::span<const double> normal_speed, std::span<const double> sign_field,
    std::span<double> rate) {
  detail::validate_grid(grid);
  const std::size_t n = level_set_count(grid);
  const std::size_t plane =
      static_cast<std::size_t>(grid.ny) * static_cast<std::size_t>(grid.nz);
  if (phi.size() != n || rate.size() != n || halo_xm.size() != plane ||
      halo_xp.size() != plane)
    throw std::invalid_argument("level-set slab sizes do not match the grid");
  if (level_set_halo_width != 1)
    throw std::logic_error("level-set halo width is not 1");
  const bool advect = mode == LevelSetMode::Advection;
  const bool normal = mode == LevelSetMode::NormalSpeed;
  const bool redistance = mode == LevelSetMode::Redistance;
  if (!advect && !normal && !redistance)
    throw std::invalid_argument("level-set mode is not recognised");
  detail::require_field(phi, n, "field", true);
  detail::require_field(velocity_x, n, "velocity", advect);
  detail::require_field(velocity_y, n, "velocity", advect);
  detail::require_field(velocity_z, n, "velocity", advect);
  detail::require_field(normal_speed, n, "normal speed", normal);
  detail::require_field(sign_field, n, "sign field", redistance);
  detail::require_field(halo_xm, plane, "halo", true);
  detail::require_field(halo_xp, plane, "halo", true);
  for (int k = 0; k < grid.nz; ++k) {
    for (int j = 0; j < grid.ny; ++j) {
      for (int i = 0; i < grid.nx; ++i) {
        const std::size_t p =
            static_cast<std::size_t>(i + grid.nx * (j + grid.ny * k));
        detail::rate_at(grid, phi, halo_xm, halo_xp, mode, velocity_x, velocity_y,
                        velocity_z, normal_speed, sign_field, i, j, k, rate[p]);
        if (!std::isfinite(rate[p]))
          throw std::domain_error("nonfinite level-set rate");
      }
    }
  }
}

/// Periodic rate. The x-halos are the owned edge planes.
inline void level_set_rate(const LevelSetGrid &grid, std::span<const double> phi,
                           LevelSetMode mode, std::span<const double> velocity_x,
                           std::span<const double> velocity_y,
                           std::span<const double> velocity_z,
                           std::span<const double> normal_speed,
                           std::span<const double> sign_field,
                           std::span<double> rate) {
  detail::validate_grid(grid);
  const std::size_t plane =
      static_cast<std::size_t>(grid.ny) * static_cast<std::size_t>(grid.nz);
  std::vector<double> xm(plane);
  std::vector<double> xp(plane);
  if (phi.size() == level_set_count(grid))
    detail::fill_periodic_x_halos(grid, phi, xm, xp);
  level_set_rate_x_slab(grid, phi, xm, xp, mode, velocity_x, velocity_y, velocity_z,
                        normal_speed, sign_field, rate);
}

/// One SSP-RK2 step. A throw leaves `phi` unchanged.
[[nodiscard]] inline LevelSetStep advance_level_set(
    const LevelSetGrid &grid, std::span<double> phi, LevelSetMode mode,
    std::span<const double> velocity_x, std::span<const double> velocity_y,
    std::span<const double> velocity_z, std::span<const double> normal_speed,
    std::span<const double> sign_field, double dt) {
  if (!(dt > 0.0) || !std::isfinite(dt))
    throw std::invalid_argument("level-set time step must be positive");
  const std::size_t n = level_set_count(grid);
  std::vector<double> held(phi.begin(), phi.end());
  std::vector<double> stage(n);
  std::vector<double> rate(n);
  std::vector<double> next(n);
  level_set_rate(grid, held, mode, velocity_x, velocity_y, velocity_z, normal_speed,
                 sign_field, rate);
  for (std::size_t p = 0; p < n; ++p) stage[p] = held[p] + dt * rate[p];
  level_set_rate(grid, stage, mode, velocity_x, velocity_y, velocity_z, normal_speed,
                 sign_field, rate);
  for (std::size_t p = 0; p < n; ++p) {
    next[p] = 0.5 * held[p] + 0.5 * (stage[p] + dt * rate[p]);
    if (!std::isfinite(next[p]))
      throw std::domain_error("nonfinite level-set update");
  }
  std::copy(next.begin(), next.end(), phi.begin());
  LevelSetStep out;
  out.dt = dt;
  out.negative_volume = level_set_negative_centroid(grid, phi).volume;
  return out;
}

/// Sussman redistance. `sign_field` is the field whose zero contour is kept.
[[nodiscard]] inline LevelSetStep
redistance_level_set(const LevelSetGrid &grid, std::span<double> phi,
                     std::span<const double> sign_field, int substeps, double cfl) {
  if (substeps < 1) throw std::invalid_argument("level-set redistance needs a step");
  const double dt = level_set_stable_dt(grid, 1.0, cfl);
  // Freeze the sign. `phi` may alias `sign_field`.
  const std::vector<double> frozen(sign_field.begin(), sign_field.end());
  LevelSetStep last;
  for (int s = 0; s < substeps; ++s) {
    last = advance_level_set(grid, phi, LevelSetMode::Redistance, {}, {}, {}, {},
                             frozen, dt);
  }
  last.dt = dt;
  return last;
}

} // namespace pfc::field::level_set
