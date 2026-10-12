// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file field_multigrid_kernels.hpp
 * @brief Point kernels of the matrix-free field multigrid (host and device).
 *
 * One level is a padded cell-centred grid (storage halo 1, x fastest) with the
 * operator
 *
 *     (A u)_i = m_i [ c_i u_i + sum_axis ( Dp_i (u_i - u_{i+e}) + Dp_{i-e} (u_i -
 * u_{i-e}) ) / h^2 ]
 *               + (1 - m_i) u_i,
 *
 * where Dp_i is the coefficient of the face between cell i and i + e (stored at
 * cell i, so the minus face reads the halo) and m is 1 for a free cell and 0 for
 * a pinned (identity) cell. A pinned neighbour is an eliminated Dirichlet value:
 * the free row couples to u_{i+-e} only through m_{i+-e} u_{i+-e}, so A is
 * symmetric. The mask halo must be current. These functions are shared by the host
 * loops in field_multigrid.hpp and the vendor kernels in field_multigrid_gpu.inc.
 */

#include <cstddef>

#include <openpfc/kernel/data/host_device.hpp>

namespace pfc::solvers::fmg {

/// Geometry of one level: owned extents, padded pitch, active axes, 1/h^2 and
/// the global index of the first owned cell (for the red-black colouring).
struct Level {
  int n[3] = {1, 1, 1};
  int pitch[3] = {3, 3, 3}; ///< padded extents n + 2
  int active[3] = {0, 0, 0};
  int low[3] = {0, 0, 0};
  double ih2[3] = {0, 0, 0};
  /// 1: every active axis is periodic, owned by this rank alone and of even extent.
  /// Values that are read through halos are then written to their periodic images
  /// as well (`store`), so the level needs no halo exchange; the even extent keeps
  /// the red-black colouring consistent across the wrap.
  int wrap = 0;
};

/// Coefficient arrays of one level (padded, halo exchanged where noted).
struct Coefficients {
  const double *c = nullptr;  ///< mass/reaction term, owned cells
  const double *dx = nullptr; ///< +x face coefficient (halo needed)
  const double *dy = nullptr; ///< +y face coefficient (halo needed)
  const double *dz = nullptr; ///< +z face coefficient (halo needed)
  const double *m = nullptr;  ///< 1 free, 0 pinned, owned cells
};

OPENPFC_HD inline std::size_t index(const Level &L, int i, int j, int k) {
  return static_cast<std::size_t>(i + 1) +
         static_cast<std::size_t>(L.pitch[0]) *
             (static_cast<std::size_t>(j + 1) +
              static_cast<std::size_t>(L.pitch[1]) *
                  static_cast<std::size_t>(k + 1));
}

OPENPFC_HD inline std::size_t owned_count(const Level &L) {
  return static_cast<std::size_t>(L.n[0]) * L.n[1] * L.n[2];
}

OPENPFC_HD inline void owned_ijk(const Level &L, std::size_t t, int &i, int &j,
                                 int &k) {
  i = static_cast<int>(t % static_cast<std::size_t>(L.n[0]));
  j = static_cast<int>((t / static_cast<std::size_t>(L.n[0])) %
                       static_cast<std::size_t>(L.n[1]));
  k = static_cast<int>(t / (static_cast<std::size_t>(L.n[0]) * L.n[1]));
}

/// u(i, j, k) = v, and on a `wrap` level also at the periodic images of a
/// boundary cell in the halo (faces, edges and corners).
OPENPFC_HD inline void store(const Level &L, double *u, int i, int j, int k,
                             double v) {
  u[index(L, i, j, k)] = v;
  if (!L.wrap) return;
  const int c[3] = {i, j, k};
  int alt[3][2], na[3];
  for (int a = 0; a < 3; ++a) {
    alt[a][0] = c[a];
    na[a] = 1;
    if (!L.active[a]) continue;
    if (c[a] == 0)
      alt[a][na[a]++] = L.n[a];
    else if (c[a] == L.n[a] - 1)
      alt[a][na[a]++] = -1;
  }
  for (int z = 0; z < na[2]; ++z)
    for (int y = 0; y < na[1]; ++y)
      for (int x = 0; x < na[0]; ++x)
        if (x || y || z) u[index(L, alt[0][x], alt[1][y], alt[2][z])] = v;
}

/// Off-diagonal sum sum_nb w_nb u_nb and diagonal c + sum_nb w_nb of a free row.
OPENPFC_HD inline void row(const Level &L, const Coefficients &C, const double *u,
                           int i, int j, int k, double &off, double &diag) {
  const std::size_t s = index(L, i, j, k);
  off = 0.0;
  diag = C.c[s];
  const double *D[3] = {C.dx, C.dy, C.dz};
  const std::size_t stride[3] = {1, static_cast<std::size_t>(L.pitch[0]),
                                 static_cast<std::size_t>(L.pitch[0]) * L.pitch[1]};
  for (int a = 0; a < 3; ++a) {
    if (!L.active[a]) continue;
    // Pinned neighbours are eliminated Dirichlet values: they keep their weight in
    // the diagonal but do not couple, so A is symmetric (free block SPD, pinned
    // rows identity).
    const double wp = D[a][s] * L.ih2[a];
    const double wm = D[a][s - stride[a]] * L.ih2[a];
    off += wp * C.m[s + stride[a]] * u[s + stride[a]] +
           wm * C.m[s - stride[a]] * u[s - stride[a]];
    diag += wp + wm;
  }
}

/// (A u) at an owned cell; u halos current.
OPENPFC_HD inline double apply(const Level &L, const Coefficients &C,
                               const double *u, int i, int j, int k) {
  const std::size_t s = index(L, i, j, k);
  if (C.m[s] == 0.0) return u[s];
  double off, diag;
  row(L, C, u, i, j, k, off, diag);
  return diag * u[s] - off;
}

/// Gauss-Seidel update of an owned cell of the given colour (parity of the
/// global index sum); other cells are left unchanged. u halos current.
OPENPFC_HD inline void smooth(const Level &L, const Coefficients &C, const double *f,
                              double *u, int i, int j, int k, int colour) {
  if (((L.low[0] + i + L.low[1] + j + L.low[2] + k) & 1) != colour) return;
  const std::size_t s = index(L, i, j, k);
  if (C.m[s] == 0.0) {
    store(L, u, i, j, k, f[s]);
    return;
  }
  double off, diag;
  row(L, C, u, i, j, k, off, diag);
  store(L, u, i, j, k, (f[s] + off) / diag);
}

/// r = f - A u at an owned cell (stored with its periodic images on a wrap level).
OPENPFC_HD inline void residual(const Level &L, const Coefficients &C,
                                const double *f, const double *u, double *r, int i,
                                int j, int k) {
  store(L, r, i, j, k, f[index(L, i, j, k)] - apply(L, C, u, i, j, k));
}

/// Coarse coefficients at coarse owned cell (I, J, K) from the fine level:
/// c and m averaged/maxed over the children, face coefficients averaged over the
/// fine faces that make up the coarse +face.
OPENPFC_HD inline void restrict_coefficients(const Level &F, const Coefficients &Cf,
                                             const Level &G, double *c, double *dx,
                                             double *dy, double *dz, double *m,
                                             int I, int J, int K) {
  const int r[3] = {F.active[0] ? 2 : 1, F.active[1] ? 2 : 1, F.active[2] ? 2 : 1};
  double cs = 0.0, ms = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
  int nx = 0, ny = 0, nz = 0;
  for (int c2 = 0; c2 < r[2]; ++c2)
    for (int b = 0; b < r[1]; ++b)
      for (int a = 0; a < r[0]; ++a) {
        const int i = r[0] * I + a, j = r[1] * J + b, k = r[2] * K + c2;
        const std::size_t s = index(F, i, j, k);
        cs += Cf.c[s];
        ms = ms > Cf.m[s] ? ms : Cf.m[s];
        if (a == r[0] - 1) {
          fx += Cf.dx[s];
          ++nx;
        }
        if (b == r[1] - 1) {
          fy += Cf.dy[s];
          ++ny;
        }
        if (c2 == r[2] - 1) {
          fz += Cf.dz[s];
          ++nz;
        }
      }
  const std::size_t S = index(G, I, J, K);
  c[S] = cs / (r[0] * r[1] * r[2]);
  m[S] = ms;
  dx[S] = fx / nx;
  dy[S] = fy / ny;
  dz[S] = fz / nz;
}

/// Coarse right-hand side at (I, J, K): the adjoint of `prolong_add` divided by
/// the number of children (weights 1/8, 3/8, 3/8, 1/8 per active axis over fine
/// cells 2I-1 .. 2I+2), with pinned fine cells weighted by zero. Fine residual
/// halos current (Full). Pinned coarse cells get zero.
OPENPFC_HD inline void restrict_residual(const Level &F, const double *rf,
                                         const double *mf, const Level &G,
                                         const double *mg, double *fg, int I, int J,
                                         int K) {
  const double w[4] = {0.125, 0.375, 0.375, 0.125};
  const int C[3] = {I, J, K};
  int hi[3];
  for (int a = 0; a < 3; ++a) hi[a] = F.active[a] ? 4 : 1;
  double sum = 0.0;
  for (int c2 = 0; c2 < hi[2]; ++c2) {
    const int k = F.active[2] ? 2 * C[2] - 1 + c2 : C[2];
    const double wz = F.active[2] ? w[c2] : 1.0;
    for (int b = 0; b < hi[1]; ++b) {
      const int j = F.active[1] ? 2 * C[1] - 1 + b : C[1];
      const double wy = F.active[1] ? w[b] : 1.0;
      for (int a = 0; a < hi[0]; ++a) {
        const int i = F.active[0] ? 2 * C[0] - 1 + a : C[0];
        const double wx = F.active[0] ? w[a] : 1.0;
        const std::size_t s = index(F, i, j, k);
        sum += wx * wy * wz * mf[s] * rf[s];
      }
    }
  }
  const std::size_t S = index(G, I, J, K);
  fg[S] = mg[S] == 0.0 ? 0.0 : sum;
}

/// Fine correction at (i, j, k) from the coarse level by cell-centred bilinear
/// (trilinear) interpolation: weight 3/4 for the parent, 1/4 for the neighbour
/// on the child's side, per active axis. Coarse u halos current (Full).
/// Added to the fine u at free cells only.
OPENPFC_HD inline void prolong_add(const Level &F, const double *mf, double *uf,
                                   const Level &G, const double *ug, int i, int j,
                                   int k) {
  const std::size_t s = index(F, i, j, k);
  if (mf[s] == 0.0) return;
  const int f[3] = {i, j, k};
  int P[3], d[3];
  for (int a = 0; a < 3; ++a) {
    if (F.active[a]) {
      P[a] = f[a] >> 1;
      d[a] = (f[a] & 1) ? 1 : -1;
    } else {
      P[a] = f[a];
      d[a] = 0;
    }
  }
  double v = 0.0;
  for (int c2 = 0; c2 < 2; ++c2) {
    if (c2 && !F.active[2]) continue;
    const double wz = F.active[2] ? (c2 ? 0.25 : 0.75) : 1.0;
    for (int b = 0; b < 2; ++b) {
      if (b && !F.active[1]) continue;
      const double wy = F.active[1] ? (b ? 0.25 : 0.75) : 1.0;
      for (int a = 0; a < 2; ++a) {
        if (a && !F.active[0]) continue;
        const double wx = F.active[0] ? (a ? 0.25 : 0.75) : 1.0;
        v += wx * wy * wz *
             ug[index(G, P[0] + a * d[0], P[1] + b * d[1], P[2] + c2 * d[2])];
      }
    }
  }
  store(F, uf, i, j, k, uf[s] + v);
}

} // namespace pfc::solvers::fmg
