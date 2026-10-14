// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file nodal_multigrid_kernels.hpp
 * @brief Point kernels of the matrix-free nodal Q1 multigrid (host and device).
 *
 * One level is a periodic node grid with owned extents n, a global offset low
 * and padded storage (x fastest). Vectors and stored stencils carry one halo
 * layer, element coefficients two. Element e is indexed by its lowest node, so
 * node i touches elements i - s, s in {0,1}^3.
 *
 * The operator row at node i is sum_o w_o(i) u_{i+o} over the 27 offsets
 * o in {-1,0,1}^3. On an element level
 *
 *     w_o(i) = Ke(o) * sum_{e shares i and i+o} a_e + sigma * Me(o) * count(o),
 *
 * where Ke(o) and Me(o) are the Q1 element stiffness and mass entries between
 * two nodes of one element that differ by o (they do not depend on which
 * element), and count(o) = prod_d (o_d == 0 ? 2 : 1) is the number of elements
 * sharing the two nodes. On a stencil level the 27 weights are stored per node.
 * These functions are shared by the host loops in nodal_multigrid.hpp and the
 * vendor kernels in nodal_multigrid_gpu.inc.
 */

#include <cstddef>
#include <cstdint>

#include <openpfc/kernel/data/host_device.hpp>

namespace pfc::solvers::nmg {

inline constexpr int kVectorHalo = 1;
inline constexpr int kCoefficientHalo = 2;

/// Geometry and constant element data of one level (passed by value to kernels).
struct Level {
  int n[3] = {1, 1, 1};   ///< owned extents
  int low[3] = {0, 0, 0}; ///< global index of the first owned node
  int gn[3] = {1, 1, 1};  ///< global extents
  int stencil = 0;        ///< 0: element coefficients, 1: stored 27-point stencil
  double sigma = 0.0;
  double ke[27] = {};   ///< element stiffness entry per offset (unit coefficient)
  double mass[27] = {}; ///< assembled consistent mass per offset (sigma excluded)
};

/// Operator data of one level.
struct Operator {
  const double *a = nullptr; ///< element coefficients, halo 2 (element level)
  const double *s = nullptr; ///< 27 stencil planes, halo 1 (stencil level)
  std::size_t plane = 0;     ///< padded size of one stencil plane
};

OPENPFC_HD inline int offset_index(int ox, int oy, int oz) {
  return (ox + 1) + 3 * (oy + 1) + 9 * (oz + 1);
}

OPENPFC_HD inline std::size_t padded_size(const Level &L, int H) {
  return static_cast<std::size_t>(L.n[0] + 2 * H) * (L.n[1] + 2 * H) *
         (L.n[2] + 2 * H);
}

OPENPFC_HD inline std::size_t index(const Level &L, int H, int i, int j, int k) {
  const std::size_t px = static_cast<std::size_t>(L.n[0] + 2 * H);
  const std::size_t py = static_cast<std::size_t>(L.n[1] + 2 * H);
  return static_cast<std::size_t>(i + H) +
         px * (static_cast<std::size_t>(j + H) +
               py * static_cast<std::size_t>(k + H));
}

OPENPFC_HD inline std::size_t vindex(const Level &L, int i, int j, int k) {
  return index(L, kVectorHalo, i, j, k);
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

/// Fill `ke` and `mass` from the spacing (unit coefficient, assembled mass).
inline void set_element_data(Level &L, const double h[3]) {
  const double V = h[0] * h[1] * h[2];
  for (int oz = -1; oz <= 1; ++oz)
    for (int oy = -1; oy <= 1; ++oy)
      for (int ox = -1; ox <= 1; ++ox) {
        const int o[3] = {ox, oy, oz};
        double k = 0.0, m = V;
        for (int d = 0; d < 3; ++d) {
          double t = V / (h[d] * h[d]) * (o[d] == 0 ? 1.0 : -1.0);
          for (int e = 0; e < 3; ++e)
            if (e != d) t *= (o[e] == 0 ? 1.0 / 3.0 : 1.0 / 6.0);
          k += t;
          m *= (o[d] == 0 ? 2.0 / 3.0 : 1.0 / 6.0);
        }
        L.ke[offset_index(ox, oy, oz)] = k;
        L.mass[offset_index(ox, oy, oz)] = m;
      }
}

/// The 27 row weights of an element level from the 8 coefficients around a node,
/// a[sz][sy][sx] being the coefficient of element (i - sx, j - sy, k - sz).
OPENPFC_HD inline void element_weights(const Level &L, const double a[2][2][2],
                                       double w[27]) {
  // Sum over the elements that contain both i and i + o: along an axis o_d = -1
  // selects s_d = 1, o_d = +1 selects s_d = 0 and o_d = 0 both.
  double bx[3][2][2];
  for (int sz = 0; sz < 2; ++sz)
    for (int sy = 0; sy < 2; ++sy) {
      bx[0][sz][sy] = a[sz][sy][1];
      bx[1][sz][sy] = a[sz][sy][0] + a[sz][sy][1];
      bx[2][sz][sy] = a[sz][sy][0];
    }
  double by[3][3][2];
  for (int ox = 0; ox < 3; ++ox)
    for (int sz = 0; sz < 2; ++sz) {
      by[0][ox][sz] = bx[ox][sz][1];
      by[1][ox][sz] = bx[ox][sz][0] + bx[ox][sz][1];
      by[2][ox][sz] = bx[ox][sz][0];
    }
  for (int oy = 0; oy < 3; ++oy)
    for (int ox = 0; ox < 3; ++ox) {
      const double s0 = by[oy][ox][0], s1 = by[oy][ox][1];
      const double sum[3] = {s1, s0 + s1, s0};
      for (int oz = 0; oz < 3; ++oz) {
        const int o = ox + 3 * oy + 9 * oz;
        w[o] = L.ke[o] * sum[oz] + L.sigma * L.mass[o];
      }
    }
}

/// The 27 row weights at owned or halo node (i, j, k).
OPENPFC_HD inline void weights(const Level &L, const Operator &A, int i, int j,
                               int k, double w[27]) {
  if (L.stencil) {
    const std::size_t s = vindex(L, i, j, k);
    for (int o = 0; o < 27; ++o)
      w[o] = A.s[static_cast<std::size_t>(o) * A.plane + s];
    return;
  }
  double a[2][2][2];
  for (int sz = 0; sz < 2; ++sz)
    for (int sy = 0; sy < 2; ++sy)
      for (int sx = 0; sx < 2; ++sx)
        a[sz][sy][sx] = A.a[index(L, kCoefficientHalo, i - sx, j - sy, k - sz)];
  element_weights(L, a, w);
}

/// (A u) at node (i, j, k); u halos current.
OPENPFC_HD inline double apply(const Level &L, const Operator &A, const double *u,
                               int i, int j, int k) {
  double w[27];
  weights(L, A, i, j, k, w);
  double y = 0.0;
  for (int oz = -1; oz <= 1; ++oz)
    for (int oy = -1; oy <= 1; ++oy)
      for (int ox = -1; ox <= 1; ++ox)
        y += w[offset_index(ox, oy, oz)] * u[vindex(L, i + ox, j + oy, k + oz)];
  return y;
}

OPENPFC_HD inline double diagonal(const Level &L, const Operator &A, int i, int j,
                                  int k) {
  if (L.stencil) return A.s[13 * A.plane + vindex(L, i, j, k)];
  double w[27];
  weights(L, A, i, j, k, w);
  return w[13];
}

/// r = f - A u (zero_u: r = f).
OPENPFC_HD inline void residual(const Level &L, const Operator &A, const double *f,
                                const double *u, double *r, int i, int j, int k,
                                int zero_u) {
  const std::size_t s = vindex(L, i, j, k);
  r[s] = zero_u ? f[s] : f[s] - apply(L, A, u, i, j, k);
}

/// One Chebyshev step: d = c1 d + c2 D^{-1} (f - A u) (zero_u: A u = 0).
OPENPFC_HD inline void chebyshev(const Level &L, const Operator &A, const double *f,
                                 const double *u, const double *invd, double *d,
                                 double c1, double c2, int i, int j, int k,
                                 int zero_u) {
  const std::size_t s = vindex(L, i, j, k);
  const double r = zero_u ? f[s] : f[s] - apply(L, A, u, i, j, k);
  d[s] = c1 * d[s] + c2 * invd[s] * r;
}

/// Gauss-Seidel update at nodes whose global index parity equals `colour`
/// (bit 0: x, bit 1: y, bit 2: z). Same-colour nodes never share a row.
OPENPFC_HD inline void gauss_seidel(const Level &L, const Operator &A,
                                    const double *f, const double *invd, double *u,
                                    int colour, int i, int j, int k) {
  const int c = ((L.low[0] + i) & 1) | (((L.low[1] + j) & 1) << 1) |
                (((L.low[2] + k) & 1) << 2);
  if (c != colour) return;
  const std::size_t s = vindex(L, i, j, k);
  u[s] += invd[s] * (f[s] - apply(L, A, u, i, j, k));
}

/// Interpolation weight of fine offset delta in {-1,0,1} along one axis.
OPENPFC_HD inline double transfer_weight(int delta) {
  return delta == 0 ? 1.0 : 0.5;
}

/// Coarse value at coarse region node (I, J, K): sum_delta w(delta) r_f(2I + delta)
/// (R = P^T). The region starts at coarse local `off`; fine r halos current.
OPENPFC_HD inline void restrict_residual(const Level &F, const double *rf,
                                         const Level &G, double *fg,
                                         const int off[3], int I, int J, int K) {
  double sum = 0.0;
  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx)
        sum += transfer_weight(dx) * transfer_weight(dy) * transfer_weight(dz) *
               rf[vindex(F, 2 * I + dx, 2 * J + dy, 2 * K + dz)];
  fg[vindex(G, I + off[0], J + off[1], K + off[2])] = sum;
}

/// u_f(i) += (P u_c)(i): an even fine index takes its coarse node, an odd one the
/// mean of its two neighbours. Coarse u halos current; region offset as above.
OPENPFC_HD inline void prolong_add(const Level &F, double *uf, const Level &G,
                                   const double *ug, const int off[3], int i, int j,
                                   int k) {
  const int f[3] = {i, j, k};
  int lo[3], cnt[3];
  for (int d = 0; d < 3; ++d) {
    lo[d] = (f[d] >> 1) + off[d];
    cnt[d] = (f[d] & 1) ? 2 : 1;
  }
  const double wz = cnt[2] == 2 ? 0.5 : 1.0, wy = cnt[1] == 2 ? 0.5 : 1.0,
               wx = cnt[0] == 2 ? 0.5 : 1.0;
  double v = 0.0;
  for (int c = 0; c < cnt[2]; ++c)
    for (int b = 0; b < cnt[1]; ++b)
      for (int a = 0; a < cnt[0]; ++a)
        v += wx * wy * wz * ug[vindex(G, lo[0] + a, lo[1] + b, lo[2] + c)];
  uf[vindex(F, i, j, k)] += v;
}

/// Rediscretized coarse element coefficient: the mean of its 8 children.
OPENPFC_HD inline void restrict_coefficient(const Level &F, const double *af,
                                            const Level &G, double *ag,
                                            const int off[3], int I, int J, int K) {
  double sum = 0.0;
  for (int c = 0; c < 2; ++c)
    for (int b = 0; b < 2; ++b)
      for (int a = 0; a < 2; ++a)
        sum += af[index(F, kCoefficientHalo, 2 * I + a, 2 * J + b, 2 * K + c)];
  ag[index(G, kCoefficientHalo, I + off[0], J + off[1], K + off[2])] = 0.125 * sum;
}

/// Galerkin coarse stencil at coarse region node (I, J, K):
/// S_O(I) = sum_delta w(delta) sum_o A_o(2I + delta) w(delta + o - 2 O),
/// over fine rows 2I + delta (fine operator halos current).
OPENPFC_HD inline void galerkin(const Level &F, const Operator &Af, const Level &G,
                                double *sg, std::size_t gplane, const int off[3],
                                int I, int J, int K) {
  double acc[27];
  for (int o = 0; o < 27; ++o) acc[o] = 0.0;
  double w[27];
  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        const double wr =
            transfer_weight(dx) * transfer_weight(dy) * transfer_weight(dz);
        weights(F, Af, 2 * I + dx, 2 * J + dy, 2 * K + dz, w);
        const int dl[3] = {dx, dy, dz};
        for (int oz = -1; oz <= 1; ++oz)
          for (int oy = -1; oy <= 1; ++oy)
            for (int ox = -1; ox <= 1; ++ox) {
              const double a = wr * w[offset_index(ox, oy, oz)];
              if (a == 0.0) continue;
              const int t[3] = {dl[0] + ox, dl[1] + oy, dl[2] + oz};
              // Per axis: the coarse offsets O with |t - 2 O| <= 1, and weight.
              int O[3][2], nO[3];
              double W[3][2];
              for (int d = 0; d < 3; ++d) {
                nO[d] = 0;
                for (int c = -1; c <= 1; ++c) {
                  const int dd = t[d] - 2 * c;
                  if (dd >= -1 && dd <= 1) {
                    O[d][nO[d]] = c;
                    W[d][nO[d]++] = transfer_weight(dd);
                  }
                }
              }
              for (int c = 0; c < nO[2]; ++c)
                for (int b = 0; b < nO[1]; ++b)
                  for (int q = 0; q < nO[0]; ++q)
                    acc[offset_index(O[0][q], O[1][b], O[2][c])] +=
                        a * W[0][q] * W[1][b] * W[2][c];
            }
      }
  const std::size_t S = vindex(G, I + off[0], J + off[1], K + off[2]);
  for (int o = 0; o < 27; ++o) sg[static_cast<std::size_t>(o) * gplane + S] = acc[o];
}

/// Deterministic value in [-1, 1] from a global node index (decomposition
/// independent start vector of the eigenvalue estimate).
OPENPFC_HD inline double hash_value(int gi, int gj, int gk) {
  std::uint64_t x =
      static_cast<std::uint64_t>(static_cast<std::uint32_t>(gi)) |
      (static_cast<std::uint64_t>(static_cast<std::uint32_t>(gj)) << 21) |
      (static_cast<std::uint64_t>(static_cast<std::uint32_t>(gk)) << 42);
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  x ^= x >> 31;
  return static_cast<double>(x >> 11) * (2.0 / 9007199254740992.0) - 1.0;
}

} // namespace pfc::solvers::nmg
