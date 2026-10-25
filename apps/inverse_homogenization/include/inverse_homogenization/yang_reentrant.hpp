// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file yang_reentrant.hpp
 * @brief Yang 2015 design A3 3-D re-entrant honeycomb (OpenPFC #43).
 *
 * Primary source: Yang, Harrysson, West & Cormier, Int. J. Solids Struct.
 * 69–70 (2015) 475–490. Adjacent 2-D re-entrant layers are shifted by half
 * a cell. Not the through-going-pillar surrogate of #36.
 *
 * Cell box from Yang §4.1.2: 2(H − L cos θ) × 2 L sin θ × 2 L sin θ.
 * θ is the angle between a vertical strut and a re-entrant strut. Members
 * have square cross-section of side t. Inward V's (not outward convex
 * hexagons). Second story offset H − 2 L cos θ.
 */

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

#include <openpfc/kernel/data/grid_field.hpp>

namespace pfc::apps::inverse {

struct YangA3 {
  static constexpr double H_mm = 7.74;
  static constexpr double L_mm = 3.78;
  static constexpr double t_mm = 0.80;
  static constexpr double theta_deg = 45.0;
  /// Yang Table 1 A3 published relative density.
  static constexpr double published_rel_density = 0.2330;
  /// Sign/mechanism expectation, not a voxel fit target.
  static constexpr double published_nu_zx = -1.90;
  /// Frozen before any Poisson evaluation (published/2 .. published*1.5).
  static constexpr double density_gate_lo = 0.113;
  static constexpr double density_gate_hi = 0.353;

  [[nodiscard]] static double theta_rad() noexcept {
    return theta_deg * 3.14159265358979323846 / 180.0;
  }
  [[nodiscard]] static double sin_th() noexcept { return std::sin(theta_rad()); }
  [[nodiscard]] static double cos_th() noexcept { return std::cos(theta_rad()); }
  [[nodiscard]] static double dx() noexcept { return L_mm * sin_th(); }
  [[nodiscard]] static double dz() noexcept { return L_mm * cos_th(); }
  [[nodiscard]] static double Lx() noexcept { return 2.0 * dx(); }
  [[nodiscard]] static double Ly() noexcept { return Lx(); }
  /// Yang 2015 bounding box in z, not H + 2 L cos θ.
  [[nodiscard]] static double Lz() noexcept { return 2.0 * (H_mm - dz()); }
  [[nodiscard]] static double H_over_L() noexcept { return H_mm / L_mm; }
  [[nodiscard]] static double t_over_L() noexcept { return t_mm / L_mm; }
  /// Next 2-D story on the same layer family (overlap H − Lz = 2 L cos θ − H).
  [[nodiscard]] static double story_shift() noexcept { return H_mm - 2.0 * dz(); }
  [[nodiscard]] static int nz_for_nx(int nx) noexcept {
    return std::max(
        1, static_cast<int>(std::lround(static_cast<double>(nx) * Lz() / Lx())));
  }
  /// Wang 2016 eq. (4) overlap-corrected density for this topology.
  [[nodiscard]] static double wang_rel_density() noexcept {
    const double a = H_over_L();
    const double tl = t_over_L();
    const double s = sin_th();
    const double c = cos_th();
    const double num = a + 4.0 - ((3.0 + c) / s) * tl;
    const double den = 2.0 * (a - c) * s * s;
    return num * tl * tl / den;
  }
};

struct YangSeg {
  double ax, ay, az, bx, by, bz;
  int layer;
  char kind;  // 'H' vertical, 'L' re-entrant
  char plane; // 'x' xz-layer, 'y' yz-layer
};

struct YangLedger {
  int n_H_drawn{};
  int n_L_drawn{};
  int n_H_unique{};
  int n_L_unique{};
  int n_xz_layers{};
  int n_yz_layers{};
  /// Wang unit-cell drawing (4 verticals + 16 obliques).
  static constexpr int n_H_source = 4;
  static constexpr int n_L_source = 16;
};

inline void yang_add_seg(std::vector<YangSeg> &s, double ax, double ay, double az,
                         double bx, double by, double bz, int layer, char kind,
                         char plane) {
  s.push_back(YangSeg{ax, ay, az, bx, by, bz, layer, kind, plane});
}

/// One 2-D re-entrant bowtie in the x–z plane: inward V's, verticals of H.
inline void yang_add_xz_layer(std::vector<YangSeg> &s, double y0, double sx,
                              double sz, int layer) {
  const double dx = YangA3::dx();
  const double cz = YangA3::dz();
  const double H = YangA3::H_mm;
  const double Lx = YangA3::Lx();
  const double zbot = sz;
  const double ztop = sz + H;
  const double zwb = sz + cz;
  const double zwt = sz + H - cz;
  auto X = [&](double x) { return sx + x; };
  yang_add_seg(s, X(0.0), y0, zbot, X(0.0), y0, ztop, layer, 'H', 'x');
  yang_add_seg(s, X(Lx), y0, zbot, X(Lx), y0, ztop, layer, 'H', 'x');
  yang_add_seg(s, X(0.0), y0, zbot, X(dx), y0, zwb, layer, 'L', 'x');
  yang_add_seg(s, X(Lx), y0, zbot, X(dx), y0, zwb, layer, 'L', 'x');
  yang_add_seg(s, X(0.0), y0, ztop, X(dx), y0, zwt, layer, 'L', 'x');
  yang_add_seg(s, X(Lx), y0, ztop, X(dx), y0, zwt, layer, 'L', 'x');
}

/// Same bowtie in the y–z plane.
inline void yang_add_yz_layer(std::vector<YangSeg> &s, double x0, double sy,
                              double sz, int layer) {
  const double dy = YangA3::dx();
  const double cz = YangA3::dz();
  const double H = YangA3::H_mm;
  const double Ly = YangA3::Ly();
  const double zbot = sz;
  const double ztop = sz + H;
  const double zwb = sz + cz;
  const double zwt = sz + H - cz;
  auto Y = [&](double y) { return sy + y; };
  yang_add_seg(s, x0, Y(0.0), zbot, x0, Y(0.0), ztop, layer, 'H', 'y');
  yang_add_seg(s, x0, Y(Ly), zbot, x0, Y(Ly), ztop, layer, 'H', 'y');
  yang_add_seg(s, x0, Y(0.0), zbot, x0, Y(dy), zwb, layer, 'L', 'y');
  yang_add_seg(s, x0, Y(Ly), zbot, x0, Y(dy), zwb, layer, 'L', 'y');
  yang_add_seg(s, x0, Y(0.0), ztop, x0, Y(dy), zwt, layer, 'L', 'y');
  yang_add_seg(s, x0, Y(Ly), ztop, x0, Y(dy), zwt, layer, 'L', 'y');
}

inline std::vector<YangSeg> yang_a3_skeleton() {
  std::vector<YangSeg> s;
  const double Lx = YangA3::Lx();
  const double Ly = YangA3::Ly();
  const double sz1 = YangA3::story_shift();
  // Each 2-D layer family has two z-stories (a column of H), and the
  // adjacent parallel family is half-cell shifted. Not the four-pillar
  // surrogate of #36.
  yang_add_xz_layer(s, 0.0, 0.0, 0.0, 0);
  yang_add_xz_layer(s, 0.0, 0.0, sz1, 1);
  yang_add_xz_layer(s, Ly * 0.5, Lx * 0.5, 0.0, 0);
  yang_add_xz_layer(s, Ly * 0.5, Lx * 0.5, sz1, 1);
  yang_add_yz_layer(s, 0.0, 0.0, 0.0, 0);
  yang_add_yz_layer(s, 0.0, 0.0, sz1, 1);
  yang_add_yz_layer(s, Lx * 0.5, Ly * 0.5, 0.0, 0);
  yang_add_yz_layer(s, Lx * 0.5, Ly * 0.5, sz1, 1);
  return s;
}

[[nodiscard]] inline bool yang_has_through_pillar(const std::vector<YangSeg> &s) {
  const double Lz = YangA3::Lz();
  for (const auto &e : s) {
    const double dx = e.bx - e.ax;
    const double dy = e.by - e.ay;
    const double dz = e.bz - e.az;
    if (std::abs(dx) < 1.0e-12 && std::abs(dy) < 1.0e-12 &&
        std::abs(std::abs(dz) - Lz) < 1.0e-9)
      return true;
  }
  return false;
}

[[nodiscard]] inline bool yang_in_square_prism(double px, double py, double pz,
                                               const YangSeg &e, double half) {
  const double vx = e.bx - e.ax;
  const double vy = e.by - e.ay;
  const double vz = e.bz - e.az;
  const double len = std::sqrt(vx * vx + vy * vy + vz * vz);
  if (len < 1.0e-15) return false;
  const double tx = vx / len;
  const double ty = vy / len;
  const double tz = vz / len;
  const double wx = px - e.ax;
  const double wy = py - e.ay;
  const double wz = pz - e.az;
  const double s = wx * tx + wy * ty + wz * tz;
  if (s < 0.0 || s > len) return false;
  const double rx = wx - s * tx;
  const double ry = wy - s * ty;
  const double rz = wz - s * tz;
  double ax = 1.0, ay = 0.0, az = 0.0;
  if (std::abs(tx) > 0.9) {
    ax = 0.0;
    ay = 1.0;
  }
  double n1x = ay * tz - az * ty;
  double n1y = az * tx - ax * tz;
  double n1z = ax * ty - ay * tx;
  const double n1n = std::sqrt(n1x * n1x + n1y * n1y + n1z * n1z);
  n1x /= n1n;
  n1y /= n1n;
  n1z /= n1n;
  const double n2x = ty * n1z - tz * n1y;
  const double n2y = tz * n1x - tx * n1z;
  const double n2z = tx * n1y - ty * n1x;
  const double u = rx * n1x + ry * n1y + rz * n1z;
  const double w = rx * n2x + ry * n2y + rz * n2z;
  return std::max(std::abs(u), std::abs(w)) <= half;
}

/// Cylinder test helper (not used for occupancy). Discriminates square vs capsule.
[[nodiscard]] inline bool yang_in_cylinder(double px, double py, double pz,
                                           const YangSeg &e, double half) {
  const double vx = e.bx - e.ax;
  const double vy = e.by - e.ay;
  const double vz = e.bz - e.az;
  const double vv = vx * vx + vy * vy + vz * vz;
  const double wx = px - e.ax;
  const double wy = py - e.ay;
  const double wz = pz - e.az;
  const double tt =
      (vv > 0.0) ? std::clamp((vx * wx + vy * wy + vz * wz) / vv, 0.0, 1.0) : 0.0;
  const double dx = px - (e.ax + tt * vx);
  const double dy = py - (e.ay + tt * vy);
  const double dz = pz - (e.az + tt * vz);
  return std::sqrt(dx * dx + dy * dy + dz * dz) <= half;
}

inline YangLedger yang_member_ledger(const std::vector<YangSeg> &s) {
  YangLedger L;
  auto key = [](const YangSeg &e, char kind) {
    auto wrap = [](double v, double Lcell) {
      v = std::fmod(v, Lcell);
      if (v < 0.0) v += Lcell;
      return v;
    };
    const double Lx = YangA3::Lx();
    const double Ly = YangA3::Ly();
    const double Lz = YangA3::Lz();
    double ax = wrap(e.ax, Lx), ay = wrap(e.ay, Ly), az = wrap(e.az, Lz);
    double bx = wrap(e.bx, Lx), by = wrap(e.by, Ly), bz = wrap(e.bz, Lz);
    auto rnd = [](double v) { return std::round(v * 1.0e6); };
    std::tuple<double, double, double, double, double, double, char> a{
        rnd(ax), rnd(ay), rnd(az), rnd(bx), rnd(by), rnd(bz), kind};
    std::tuple<double, double, double, double, double, double, char> b{
        rnd(bx), rnd(by), rnd(bz), rnd(ax), rnd(ay), rnd(az), kind};
    return a < b ? a : b;
  };
  std::vector<decltype(key(s.front(), 'H'))> uh, ul;
  int xz = 0, yz = 0;
  for (const auto &e : s) {
    if (e.kind == 'H') {
      ++L.n_H_drawn;
      uh.push_back(key(e, 'H'));
    } else {
      ++L.n_L_drawn;
      ul.push_back(key(e, 'L'));
    }
    if (e.plane == 'x' && e.kind == 'H') ++xz;
    if (e.plane == 'y' && e.kind == 'H') ++yz;
  }
  std::sort(uh.begin(), uh.end());
  uh.erase(std::unique(uh.begin(), uh.end()), uh.end());
  std::sort(ul.begin(), ul.end());
  ul.erase(std::unique(ul.begin(), ul.end()), ul.end());
  L.n_H_unique = static_cast<int>(uh.size());
  L.n_L_unique = static_cast<int>(ul.size());
  L.n_xz_layers = 2;
  L.n_yz_layers = 2;
  (void)xz;
  (void)yz;
  return L;
}

inline void fill_yang_a3(RealField &h, int nx, int ny, int nz) {
  const auto segs = yang_a3_skeleton();
  const double Lx = YangA3::Lx();
  const double Ly = YangA3::Ly();
  const double Lz = YangA3::Lz();
  const double half = 0.5 * YangA3::t_mm;
  const auto n = h.local_size();
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto g = h.global(i, j, k);
        const double x = (static_cast<double>(g[0]) + 0.5) / nx * Lx;
        const double y = (static_cast<double>(g[1]) + 0.5) / ny * Ly;
        const double z = (static_cast<double>(g[2]) + 0.5) / nz * Lz;
        bool solid = false;
        for (int ix = -1; ix <= 1 && !solid; ++ix)
          for (int iy = -1; iy <= 1 && !solid; ++iy)
            for (int iz = -1; iz <= 1 && !solid; ++iz) {
              const double px = x + static_cast<double>(ix) * Lx;
              const double py = y + static_cast<double>(iy) * Ly;
              const double pz = z + static_cast<double>(iz) * Lz;
              for (const auto &e : segs) {
                if (yang_in_square_prism(px, py, pz, e, half)) {
                  solid = true;
                  break;
                }
              }
            }
        h(i, j, k) = solid ? 1.0 : 0.0;
      }
    }
  }
  h.note_host_write();
}

} // namespace pfc::apps::inverse
