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

/// Point-in-cube after a Rodrigues rotation by -angle about unit axis u.
inline bool in_rotated_cube(double px, double py, double pz, double cx, double cy,
                            double cz, double half, double ux, double uy, double uz,
                            double angle) noexcept {
  const double dx = pdelta(px, cx);
  const double dy = pdelta(py, cy);
  const double dz = pdelta(pz, cz);
  const double nrm = std::sqrt(ux * ux + uy * uy + uz * uz);
  if (nrm <= 0.0) {
    return std::abs(dx) <= half && std::abs(dy) <= half && std::abs(dz) <= half;
  }
  ux /= nrm;
  uy /= nrm;
  uz /= nrm;
  const double c = std::cos(-angle);
  const double s = std::sin(-angle);
  const double omc = 1.0 - c;
  const double dot = ux * dx + uy * dy + uz * dz;
  const double cxv_x = uy * dz - uz * dy;
  const double cxv_y = uz * dx - ux * dz;
  const double cxv_z = ux * dy - uy * dx;
  const double lx = dx * c + cxv_x * s + ux * dot * omc;
  const double ly = dy * c + cxv_y * s + uy * dot * omc;
  const double lz = dz * c + cxv_z * s + uz * dot * omc;
  return std::abs(lx) <= half && std::abs(ly) <= half && std::abs(lz) <= half;
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
  const std::array<std::array<double, 2>, 4> c{
      {{{0.25, 0.25}}, {{0.75, 0.25}}, {{0.75, 0.75}}, {{0.25, 0.75}}}};
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
      {{xL, 0.00, xR, 0.00}},     {{xL, 0.50, xR, 0.50}},
      {{xL, 0.00, xSeam, 0.25}},  {{xSeam, 0.25, xL, 0.50}},
      {{xL, 0.50, xSeam, 0.75}},  {{xSeam, 0.75, xL, 1.00}},
      {{xR, 0.00, xSeamR, 0.25}}, {{xSeamR, 0.25, xR, 0.50}},
      {{xR, 0.50, xSeamR, 0.75}}, {{xSeamR, 0.75, xR, 1.00}},
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

/**
 * @brief 3-D rotating-cube seed (forward-oracle geometry, OpenPFC #31).
 *
 * Eight cubes on the octant centres of the unit cell. Neighbouring cubes
 * rotate in opposite sense about cubic-symmetric axes so they meet at
 * edge hinges. This is the 3-D analogue of `fill_rotating_squares`, not
 * an extrusion: occupancy varies in x, y, and z.
 *
 * Mechanical idea: Attard, D. & Grima, J. N., Phys. Status Solidi B 249
 * (2012) 1330–1338 (3-D rotating rigid units). The voxelisation below
 * reproduces cubes + opposite-sense rotations, not a digitised figure.
 *
 * @param half   cube half-side in the unit cell (0.20–0.22; 0.16 leaves
 *               islands in 2-D and will disconnect 3-D hinges too)
 * @param angle  rotation in radians (0.35–0.45 typical)
 */
inline void fill_rotating_cubes(RealField &h, int nx, int ny, int nz, double half,
                                double angle) {
  const auto n = h.local_size();
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto g = h.global(i, j, k);
        const double x = (static_cast<double>(g[0]) + 0.5) / static_cast<double>(nx);
        const double y = (static_cast<double>(g[1]) + 0.5) / static_cast<double>(ny);
        const double z = (static_cast<double>(g[2]) + 0.5) / static_cast<double>(nz);
        bool solid = false;
        for (int iz = 0; iz < 2 && !solid; ++iz) {
          for (int iy = 0; iy < 2 && !solid; ++iy) {
            for (int ix = 0; ix < 2; ++ix) {
              const double cx = 0.25 + 0.5 * static_cast<double>(ix);
              const double cy = 0.25 + 0.5 * static_cast<double>(iy);
              const double cz = 0.25 + 0.5 * static_cast<double>(iz);
              const double ux = ((iy + iz) % 2 == 0) ? 1.0 : -1.0;
              const double uy = ((iz + ix) % 2 == 0) ? 1.0 : -1.0;
              const double uz = ((ix + iy) % 2 == 0) ? 1.0 : -1.0;
              if (in_rotated_cube(x, y, z, cx, cy, cz, half, ux, uy, uz, angle)) {
                solid = true;
                break;
              }
            }
          }
        }
        h(i, j, k) = solid ? 1.0 : 0.0;
      }
    }
  }
  h.note_host_write();
}

inline double dist_segment_periodic_3d(double px, double py, double pz, double ax,
                                       double ay, double az, double bx, double by,
                                       double bz) noexcept {
  double best = 1.0e300;
  for (int ix = -1; ix <= 1; ++ix) {
    for (int iy = -1; iy <= 1; ++iy) {
      for (int iz = -1; iz <= 1; ++iz) {
        const double qx = px + static_cast<double>(ix);
        const double qy = py + static_cast<double>(iy);
        const double qz = pz + static_cast<double>(iz);
        const double vx = bx - ax;
        const double vy = by - ay;
        const double vz = bz - az;
        const double wx = qx - ax;
        const double wy = qy - ay;
        const double wz = qz - az;
        const double vv = vx * vx + vy * vy + vz * vz;
        const double t =
            (vv > 0.0) ? std::clamp((vx * wx + vy * wy + vz * wz) / vv, 0.0, 1.0)
                       : 0.0;
        const double dx = qx - (ax + t * vx);
        const double dy = qy - (ay + t * vy);
        const double dz = qz - (az + t * vz);
        best = std::min(best, dx * dx + dy * dy + dz * dz);
      }
    }
  }
  return std::sqrt(best);
}

struct Seg3 {
  double ax, ay, az, bx, by, bz;
};

/// One re-entrant square of side 0.5: four V-pairs + waist verticals, all
/// pointing inward. 3-D re-entrant honeycomb of Evans / Yang 2015, using
/// Gibson–Robert four-fold patterning about vertical struts.
inline void append_reentrant_square(std::vector<Seg3> &segs, double x0, double y0,
                                    double inset, double zb, double zw0, double zw1,
                                    double zt) {
  const double x1 = x0 + 0.5;
  const double y1 = y0 + 0.5;
  const double xm = x0 + 0.25;
  const double ym = y0 + 0.25;
  auto add = [&](double ax, double ay, double az, double bx, double by, double bz) {
    segs.push_back(Seg3{ax, ay, az, bx, by, bz});
  };
  // x-edges at y=y0 (inward +y) and y=y1 (inward -y)
  add(x0, y0, zb, xm, y0 + inset, zw0);
  add(xm, y0 + inset, zw0, x1, y0, zb);
  add(x0, y0, zt, xm, y0 + inset, zw1);
  add(xm, y0 + inset, zw1, x1, y0, zt);
  add(xm, y0 + inset, zw0, xm, y0 + inset, zw1);
  add(x0, y1, zb, xm, y1 - inset, zw0);
  add(xm, y1 - inset, zw0, x1, y1, zb);
  add(x0, y1, zt, xm, y1 - inset, zw1);
  add(xm, y1 - inset, zw1, x1, y1, zt);
  add(xm, y1 - inset, zw0, xm, y1 - inset, zw1);
  // y-edges at x=x0 (inward +x) and x=x1 (inward -x)
  add(x0, y0, zb, x0 + inset, ym, zw0);
  add(x0 + inset, ym, zw0, x0, y1, zb);
  add(x0, y0, zt, x0 + inset, ym, zw1);
  add(x0 + inset, ym, zw1, x0, y1, zt);
  add(x0 + inset, ym, zw0, x0 + inset, ym, zw1);
  add(x1, y0, zb, x1 - inset, ym, zw0);
  add(x1 - inset, ym, zw0, x1, y1, zb);
  add(x1, y0, zt, x1 - inset, ym, zw1);
  add(x1 - inset, ym, zw1, x1, y1, zt);
  add(x1 - inset, ym, zw0, x1 - inset, ym, zw1);
}

/**
 * @brief 3-D re-entrant honeycomb (forward oracle, OpenPFC #36).
 *
 * Vertical pillars on the half-lattice {(0,0),(0.5,0),(0,0.5),(0.5,0.5)}
 * plus re-entrant V-struts on a checkerboard of 0.5-squares. This is the
 * Evans inverted honeycomb / Yang et al. (2015) TRH, patterned with
 * four-fold symmetry about the verticals (Gibson–Robert). Not an
 * extrusion and not the eight-cube occupancy of #31.
 *
 * Frozen defaults: t=0.06, inset=0.14, z joints 0.10/0.30/0.70/0.90.
 */
inline void fill_reentrant_3d(RealField &h, int nx, int ny, int nz, double t,
                              double inset, double zb, double zw0, double zw1,
                              double zt) {
  std::vector<Seg3> segs;
  for (double x : {0.0, 0.5})
    for (double y : {0.0, 0.5}) segs.push_back(Seg3{x, y, 0.0, x, y, 1.0});
  append_reentrant_square(segs, 0.0, 0.0, inset, zb, zw0, zw1, zt);
  append_reentrant_square(segs, 0.5, 0.0, inset, zb, zw0, zw1, zt);
  append_reentrant_square(segs, 0.0, 0.5, inset, zb, zw0, zw1, zt);
  append_reentrant_square(segs, 0.5, 0.5, inset, zb, zw0, zw1, zt);
  const auto n = h.local_size();
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto g = h.global(i, j, k);
        const double x = (static_cast<double>(g[0]) + 0.5) / static_cast<double>(nx);
        const double y = (static_cast<double>(g[1]) + 0.5) / static_cast<double>(ny);
        const double z = (static_cast<double>(g[2]) + 0.5) / static_cast<double>(nz);
        double dmin = 1.0e300;
        for (const auto &s : segs)
          dmin = std::min(dmin, dist_segment_periodic_3d(x, y, z, s.ax, s.ay, s.az,
                                                         s.bx, s.by, s.bz));
        h(i, j, k) = (dmin <= t) ? 1.0 : 0.0;
      }
    }
  }
  h.note_host_write();
}

} // namespace pfc::apps::inverse
