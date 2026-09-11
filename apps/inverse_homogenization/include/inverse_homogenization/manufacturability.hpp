// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file manufacturability.hpp
 * @brief Periodic connectivity and feature-size metrics (issue #161 Stage 7).
 *
 * Single-rank dense fields (science jobs use one rank). Not AM process
 * simulation: percolation, island volume, and morphological opening loss.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <queue>
#include <vector>

#include <openpfc/kernel/data/grid_field.hpp>

namespace pfc::apps::inverse {

using RealField = pfc::data::Field<double>;

struct Manufacturability {
  int n_solid_components{0};
  int n_void_components{0};
  double island_solid_frac{0.0};
  double island_void_frac{0.0};
  bool percolate_solid_x{false};
  bool percolate_solid_y{false};
  bool percolate_solid_z{false};
  bool percolate_void_x{false};
  bool percolate_void_y{false};
  bool percolate_void_z{false};
  /// Fraction of solid removed by a 1-cell opening (erode then dilate).
  double opening_loss_r1{0.0};
  double opening_loss_r2{0.0};
  double solid_frac{0.0};
  double grey_fraction{0.0};
};

inline int m_idx(int i, int j, int k, int nx, int ny) noexcept {
  return (k * ny + j) * nx + i;
}

/// Copy a single-rank field into a dense i-fastest buffer.
inline std::vector<double> dense_from_field(const RealField &h, int nx, int ny,
                                            int nz) {
  std::vector<double> a(static_cast<std::size_t>(nx) * ny * nz, 0.0);
  const auto n = h.local_size();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) {
        const auto g = h.global(i, j, k);
        a[static_cast<std::size_t>(m_idx(g[0], g[1], g[2], nx, ny))] = h(i, j, k);
      }
  return a;
}

struct CompStats {
  int n_components{0};
  double island_frac{0.0};
  bool perc_x{false};
  bool perc_y{false};
  bool perc_z{false};
};

inline CompStats components_periodic(const std::vector<char> &mask, int nx, int ny,
                                     int nz) {
  CompStats out;
  const int N = nx * ny * nz;
  std::vector<char> seen(static_cast<std::size_t>(N), 0);
  std::vector<int> wrap_x(static_cast<std::size_t>(N), 0);
  std::vector<int> wrap_y(static_cast<std::size_t>(N), 0);
  std::vector<int> wrap_z(static_cast<std::size_t>(N), 0);
  std::vector<char> vis(static_cast<std::size_t>(N), 0);
  int solid_n = 0;
  for (int p = 0; p < N; ++p)
    if (mask[static_cast<std::size_t>(p)]) ++solid_n;
  if (solid_n == 0) {
    out.island_frac = 0.0;
    return out;
  }
  int largest = 0;
  const int di[6] = {1, -1, 0, 0, 0, 0};
  const int dj[6] = {0, 0, 1, -1, 0, 0};
  const int dk[6] = {0, 0, 0, 0, 1, -1};
  const int nnb = (nz > 1) ? 6 : 4;
  for (int seed = 0; seed < N; ++seed) {
    if (!mask[static_cast<std::size_t>(seed)] || vis[static_cast<std::size_t>(seed)])
      continue;
    ++out.n_components;
    std::fill(seen.begin(), seen.end(), 0);
    std::queue<int> q;
    vis[static_cast<std::size_t>(seed)] = 1;
    seen[static_cast<std::size_t>(seed)] = 1;
    wrap_x[static_cast<std::size_t>(seed)] = 0;
    wrap_y[static_cast<std::size_t>(seed)] = 0;
    wrap_z[static_cast<std::size_t>(seed)] = 0;
    q.push(seed);
    int count = 0;
    bool px = false, py = false, pz = false;
    while (!q.empty()) {
      const int p = q.front();
      q.pop();
      ++count;
      const int k = p / (nx * ny);
      const int rem = p - k * nx * ny;
      const int j = rem / nx;
      const int i = rem - j * nx;
      for (int n = 0; n < nnb; ++n) {
        int ii = i + di[n], jj = j + dj[n], kk = k + dk[n];
        int wx = wrap_x[static_cast<std::size_t>(p)];
        int wy = wrap_y[static_cast<std::size_t>(p)];
        int wz = wrap_z[static_cast<std::size_t>(p)];
        if (ii < 0) {
          ii += nx;
          --wx;
        } else if (ii >= nx) {
          ii -= nx;
          ++wx;
        }
        if (jj < 0) {
          jj += ny;
          --wy;
        } else if (jj >= ny) {
          jj -= ny;
          ++wy;
        }
        if (nz > 1) {
          if (kk < 0) {
            kk += nz;
            --wz;
          } else if (kk >= nz) {
            kk -= nz;
            ++wz;
          }
        } else {
          kk = 0;
        }
        const int np = m_idx(ii, jj, kk, nx, ny);
        if (!mask[static_cast<std::size_t>(np)]) continue;
        if (seen[static_cast<std::size_t>(np)]) {
          if (wrap_x[static_cast<std::size_t>(np)] != wx) px = true;
          if (wrap_y[static_cast<std::size_t>(np)] != wy) py = true;
          if (nz > 1 && wrap_z[static_cast<std::size_t>(np)] != wz) pz = true;
          continue;
        }
        seen[static_cast<std::size_t>(np)] = 1;
        vis[static_cast<std::size_t>(np)] = 1;
        wrap_x[static_cast<std::size_t>(np)] = wx;
        wrap_y[static_cast<std::size_t>(np)] = wy;
        wrap_z[static_cast<std::size_t>(np)] = wz;
        q.push(np);
      }
    }
    if (count > largest) largest = count;
    out.perc_x = out.perc_x || px;
    out.perc_y = out.perc_y || py;
    out.perc_z = out.perc_z || pz;
  }
  out.island_frac =
      1.0 - static_cast<double>(largest) / static_cast<double>(solid_n);
  return out;
}

inline std::vector<char> erode(const std::vector<char> &m, int nx, int ny, int nz) {
  std::vector<char> o = m;
  const int nnb = (nz > 1) ? 6 : 4;
  const int di[6] = {1, -1, 0, 0, 0, 0};
  const int dj[6] = {0, 0, 1, -1, 0, 0};
  const int dk[6] = {0, 0, 0, 0, 1, -1};
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        if (!m[static_cast<std::size_t>(m_idx(i, j, k, nx, ny))]) continue;
        for (int n = 0; n < nnb; ++n) {
          int ii = (i + di[n] + nx) % nx;
          int jj = (j + dj[n] + ny) % ny;
          int kk = (nz > 1) ? (k + dk[n] + nz) % nz : 0;
          if (!m[static_cast<std::size_t>(m_idx(ii, jj, kk, nx, ny))]) {
            o[static_cast<std::size_t>(m_idx(i, j, k, nx, ny))] = 0;
            break;
          }
        }
      }
  return o;
}

inline std::vector<char> dilate(const std::vector<char> &m, int nx, int ny, int nz) {
  std::vector<char> o = m;
  const int nnb = (nz > 1) ? 6 : 4;
  const int di[6] = {1, -1, 0, 0, 0, 0};
  const int dj[6] = {0, 0, 1, -1, 0, 0};
  const int dk[6] = {0, 0, 0, 0, 1, -1};
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        if (m[static_cast<std::size_t>(m_idx(i, j, k, nx, ny))]) continue;
        for (int n = 0; n < nnb; ++n) {
          int ii = (i + di[n] + nx) % nx;
          int jj = (j + dj[n] + ny) % ny;
          int kk = (nz > 1) ? (k + dk[n] + nz) % nz : 0;
          if (m[static_cast<std::size_t>(m_idx(ii, jj, kk, nx, ny))]) {
            o[static_cast<std::size_t>(m_idx(i, j, k, nx, ny))] = 1;
            break;
          }
        }
      }
  return o;
}

inline double count_ones(const std::vector<char> &m) {
  double s = 0.0;
  for (char c : m)
    if (c) s += 1.0;
  return s;
}

inline Manufacturability measure_manufacturability(const std::vector<double> &h,
                                                   int nx, int ny, int nz,
                                                   double thresh = 0.5) {
  Manufacturability m;
  const std::size_t N = h.size();
  std::vector<char> solid(N, 0), voids(N, 0);
  int grey = 0;
  for (std::size_t p = 0; p < N; ++p) {
    if (h[p] > 0.1 && h[p] < 0.9) ++grey;
    if (h[p] > thresh) solid[p] = 1;
    else voids[p] = 1;
  }
  m.grey_fraction = static_cast<double>(grey) / static_cast<double>(N);
  m.solid_frac = count_ones(solid) / static_cast<double>(N);
  const auto sc = components_periodic(solid, nx, ny, nz);
  const auto vc = components_periodic(voids, nx, ny, nz);
  m.n_solid_components = sc.n_components;
  m.n_void_components = vc.n_components;
  m.island_solid_frac = sc.island_frac;
  m.island_void_frac = vc.island_frac;
  m.percolate_solid_x = sc.perc_x;
  m.percolate_solid_y = sc.perc_y;
  m.percolate_solid_z = sc.perc_z;
  m.percolate_void_x = vc.perc_x;
  m.percolate_void_y = vc.perc_y;
  m.percolate_void_z = vc.perc_z;
  const double s0 = count_ones(solid);
  if (s0 > 0.0) {
    auto e1 = erode(solid, nx, ny, nz);
    auto o1 = dilate(e1, nx, ny, nz);
    m.opening_loss_r1 = 1.0 - count_ones(o1) / s0;
    auto e2 = erode(e1, nx, ny, nz);
    auto o2 = dilate(dilate(e2, nx, ny, nz), nx, ny, nz);
    m.opening_loss_r2 = 1.0 - count_ones(o2) / s0;
  }
  return m;
}

inline Manufacturability measure_manufacturability(const RealField &h, int nx,
                                                   int ny, int nz,
                                                   double thresh = 0.5) {
  return measure_manufacturability(dense_from_field(h, nx, ny, nz), nx, ny, nz,
                                   thresh);
}

} // namespace pfc::apps::inverse
