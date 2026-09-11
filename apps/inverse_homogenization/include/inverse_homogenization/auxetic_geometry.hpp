// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file auxetic_geometry.hpp
 * @brief Periodic 2-D seeds that *can* homogenize to C_12 < 0.
 *
 * Random Fourier noise never entered the auxetic basin (jobs 21954505–
 * 21955207). These geometries are the classical mechanisms: rotating
 * squares (Grima) and a re-entrant honeycomb. Solid is h=1 (walls /
 * squares), void is h=0.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <openpfc/kernel/data/grid_field.hpp>

namespace pfc::apps::inverse {

using RealField = pfc::data::Field<double>;

inline double wrap01(double x) noexcept {
  x -= std::floor(x);
  return x;
}

/// Periodic displacement in a unit cell, in [-0.5, 0.5].
inline double pdelta(double a, double b) noexcept {
  double d = a - b;
  d -= std::round(d);
  return d;
}

inline double dist_segment_periodic(double px, double py, double ax, double ay,
                                    double bx, double by) noexcept {
  double best = 1.0e300;
  for (int ix = -1; ix <= 1; ++ix) {
    for (int iy = -1; iy <= 1; ++iy) {
      const double qx = px + static_cast<double>(ix);
      const double qy = py + static_cast<double>(iy);
      const double vx = bx - ax;
      const double vy = by - ay;
      const double wx = qx - ax;
      const double wy = qy - ay;
      const double vv = vx * vx + vy * vy;
      const double t =
          (vv > 0.0) ? std::clamp((vx * wx + vy * wy) / vv, 0.0, 1.0) : 0.0;
      const double dx = qx - (ax + t * vx);
      const double dy = qy - (ay + t * vy);
      best = std::min(best, dx * dx + dy * dy);
    }
  }
  return std::sqrt(best);
}

inline bool in_rotated_square(double px, double py, double cx, double cy,
                              double half, double angle) noexcept {
  const double dx = pdelta(px, cx);
  const double dy = pdelta(py, cy);
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  const double lx = c * dx + s * dy;
  const double ly = -s * dx + c * dy;
  return std::abs(lx) <= half && std::abs(ly) <= half;
}

/**
 * @brief Grima rotating-square seed.
 *
 * Four squares on a square lattice, neighbouring squares rotated
 * opposite ways so they meet at thin corner hinges. Under uniaxial
 * stretch the squares rotate and the cell expands laterally.
 *
 * @param half    half-side in the unit cell. Must be large enough that
 *                neighbouring squares share a hinge (≈0.21 at 0.4 rad).
 *                0.185 left four disconnected islands (job 21955691).
 * @param angle   rotation in radians (0.40–0.50 typical)
 */
inline void fill_rotating_squares(RealField &h, int nx, int ny, double half,
                                  double angle) {
  const auto n = h.local_size();
  const std::array<std::array<double, 2>, 4> c{{{{0.25, 0.25}},
                                                {{0.75, 0.25}},
                                                {{0.75, 0.75}},
                                                {{0.25, 0.75}}}};
  const std::array<double, 4> ang{{angle, -angle, angle, -angle}};
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto g = h.global(i, j, k);
        const double x = (static_cast<double>(g[0]) + 0.5) / static_cast<double>(nx);
        const double y = (static_cast<double>(g[1]) + 0.5) / static_cast<double>(ny);
        bool solid = false;
        for (int s = 0; s < 4; ++s) {
          if (in_rotated_square(x, y, c[static_cast<std::size_t>(s)][0],
                                c[static_cast<std::size_t>(s)][1], half,
                                ang[static_cast<std::size_t>(s)])) {
            solid = true;
            break;
          }
        }
        h(i, j, k) = solid ? 1.0 : 0.0;
      }
    }
  }
  h.note_host_write();
}

/**
 * @brief Re-entrant (inverted) honeycomb walls.
 *
 * Horizontal ligaments plus inward diagonals. @p t is wall half-thickness
 * in the unit cell (0.04–0.07). @p inset is how far the waist pulls in
 * (0.12–0.20); larger inset is more re-entrant.
 */
inline void fill_reentrant_honeycomb(RealField &h, int nx, int ny, double t,
                                     double inset) {
  using Seg = std::array<double, 4>;
  // Short horizontals + diagonals that meet at the periodic x=0/1 seam.
  // A full-width bar (job 21955873) is a plate, not a re-entrant honeycomb.
  const double xL = 0.35;
  const double xR = 0.65;
  const double xSeam = 0.0;
  const double xSeamR = 1.0;
  (void)inset;
  const std::vector<Seg> segs = {
      {{xL, 0.00, xR, 0.00}},
      {{xL, 0.50, xR, 0.50}},
      {{xL, 0.00, xSeam, 0.25}},
      {{xSeam, 0.25, xL, 0.50}},
      {{xL, 0.50, xSeam, 0.75}},
      {{xSeam, 0.75, xL, 1.00}},
      {{xR, 0.00, xSeamR, 0.25}},
      {{xSeamR, 0.25, xR, 0.50}},
      {{xR, 0.50, xSeamR, 0.75}},
      {{xSeamR, 0.75, xR, 1.00}},
  };
  const auto n = h.local_size();
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto g = h.global(i, j, k);
        const double x = (static_cast<double>(g[0]) + 0.5) / static_cast<double>(nx);
        const double y = (static_cast<double>(g[1]) + 0.5) / static_cast<double>(ny);
        double dmin = 1.0e300;
        for (const auto &s : segs) {
          dmin = std::min(dmin, dist_segment_periodic(x, y, s[0], s[1], s[2], s[3]));
        }
        h(i, j, k) = (dmin <= t) ? 1.0 : 0.0;
      }
    }
  }
  h.note_host_write();
}

} // namespace pfc::apps::inverse
