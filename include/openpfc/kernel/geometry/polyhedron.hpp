// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file polyhedron.hpp
 * @brief Fixed-capacity convex polyhedron for one Cartesian cell.
 *
 * The capacity is the bound for a box cut by a single plane (at most 10
 * vertices). This is not a mesh data structure: it does not outlive the
 * cell that is being integrated.
 */

#include <openpfc/kernel/geometry/vec3.hpp>

namespace pfc::geometry {

struct Poly {
  int nv = 0;
  Vec3 v[16]{};
  int nf = 0;
  int flen[12]{};
  int fv[12][8]{};
  int cap = -1;
};

OPENPFC_HD inline double poly_volume(const Poly &poly) {
  double sum = 0;
  for (int f = 0; f < poly.nf; ++f) {
    if (poly.flen[f] < 3) continue;
    const Vec3 a = poly.v[poly.fv[f][0]];
    for (int i = 1; i + 1 < poly.flen[f]; ++i) {
      const Vec3 b = poly.v[poly.fv[f][i]];
      const Vec3 c = poly.v[poly.fv[f][i + 1]];
      sum += triple(a, b, c);
    }
  }
  return sum / 6.0;
}

OPENPFC_HD inline double poly_area(const Poly &poly, int face) {
  if (face < 0 || poly.flen[face] < 3) return 0;
  const Vec3 a = poly.v[poly.fv[face][0]];
  double area = 0;
  for (int i = 1; i + 1 < poly.flen[face]; ++i) {
    const Vec3 b = poly.v[poly.fv[face][i]];
    const Vec3 c = poly.v[poly.fv[face][i + 1]];
    area += 0.5 * norm(cross(b - a, c - a));
  }
  return area;
}

OPENPFC_HD inline void add_face(Poly &poly, const int *idx, int n) {
  if (poly.nf >= 12 || n < 3 || n > 8) return;
  poly.flen[poly.nf] = n;
  for (int i = 0; i < n; ++i) poly.fv[poly.nf][i] = idx[i];
  ++poly.nf;
}

OPENPFC_HD inline Poly poly_box(Box box) {
  Poly poly;
  const Vec3 lo = box.lo;
  const Vec3 hi = box.hi;
  for (int k = 0; k < 8; ++k) {
    poly.v[k] = {(k & 1) ? hi.x : lo.x, (k & 2) ? hi.y : lo.y,
                 (k & 4) ? hi.z : lo.z};
  }
  poly.nv = 8;
  const int faces[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4},
                           {2, 6, 7, 3}, {0, 4, 6, 2}, {1, 3, 7, 5}};
  for (int f = 0; f < 6; ++f) add_face(poly, faces[f], 4);
  return poly;
}

OPENPFC_HD inline Poly poly_tet(const Vec3 p[4]) {
  Poly poly;
  Vec3 q[4] = {p[0], p[1], p[2], p[3]};
  if (triple(q[1] - q[0], q[2] - q[0], q[3] - q[0]) < 0) {
    Vec3 swap = q[2];
    q[2] = q[3];
    q[3] = swap;
  }
  for (int i = 0; i < 4; ++i) poly.v[i] = q[i];
  poly.nv = 4;
  const int faces[4][3] = {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
  for (int f = 0; f < 4; ++f) add_face(poly, faces[f], 3);
  return poly;
}

OPENPFC_HD inline int clip_intersect(const Poly &in, Poly &out, int *cache,
                                     int *is_new, Vec3 normal, double offset,
                                     double tol, int a, int b) {
  const int i0 = a < b ? a : b;
  const int i1 = a < b ? b : a;
  int &slot = cache[i0 * 16 + i1];
  if (slot >= 0) return slot;
  const double da = dot(normal, in.v[a]);
  const double db = dot(normal, in.v[b]);
  const double denom = db - da;
  double t = absd(denom) > 0 ? (offset - da) / denom : 0;
  t = clamp_scalar(t, 0, 1);
  const Vec3 point = in.v[a] + (in.v[b] - in.v[a]) * t;
  const double tol2 = tol * tol;
  for (int i = 0; i < out.nv; ++i) {
    if (norm2(out.v[i] - point) <= tol2) {
      is_new[i] = 1;
      slot = i;
      return i;
    }
  }
  if (out.nv >= 16) return -1;
  slot = out.nv;
  out.v[out.nv] = point;
  is_new[out.nv] = 1;
  ++out.nv;
  return slot;
}

/// Keep the half-space `dot(normal, x) <= offset`. `cap` is the new face.
OPENPFC_HD inline bool clip_halfspace(const Poly &in, Vec3 normal, double offset,
                                      Poly &out) {
  out.nv = 0;
  out.nf = 0;
  out.cap = -1;
  if (in.nv <= 0) return true;
  const double scale = 1 + absd(offset) + norm(in.v[0]);
  const double tol = 1e-12 * scale;
  int inside[16];
  int ninside = 0;
  for (int i = 0; i < in.nv; ++i) {
    inside[i] = dot(normal, in.v[i]) <= offset + tol;
    ninside += inside[i];
  }
  if (ninside == in.nv) {
    out = in;
    out.cap = -1;
    return true;
  }
  if (ninside == 0) return true;
  int mapv[16];
  int is_new[16];
  for (int i = 0; i < 16; ++i) {
    mapv[i] = -1;
    is_new[i] = 0;
  }
  for (int i = 0; i < in.nv; ++i)
    if (inside[i]) {
      if (out.nv >= 16) return false;
      mapv[i] = out.nv;
      out.v[out.nv] = in.v[i];
      if (absd(dot(normal, in.v[i]) - offset) <= tol) is_new[out.nv] = 1;
      ++out.nv;
    }
  int cache[16 * 16];
  for (int i = 0; i < 16 * 16; ++i) cache[i] = -1;
  int seg_a[12], seg_b[12], nseg = 0;
  for (int f = 0; f < in.nf; ++f) {
    int nl[8];
    int nc = 0;
    for (int k = 0; k < in.flen[f]; ++k) {
      const int prev = in.fv[f][(k + in.flen[f] - 1) % in.flen[f]];
      const int cur = in.fv[f][k];
      if (inside[cur]) {
        if (!inside[prev]) {
          const int hit =
              clip_intersect(in, out, cache, is_new, normal, offset, tol, prev, cur);
          if (hit < 0 || nc >= 8) return false;
          nl[nc++] = hit;
        }
        if (nc >= 8) return false;
        nl[nc++] = mapv[cur];
      } else if (inside[prev]) {
        const int hit =
            clip_intersect(in, out, cache, is_new, normal, offset, tol, prev, cur);
        if (hit < 0 || nc >= 8) return false;
        nl[nc++] = hit;
      }
    }
    int compact[8];
    int ncompact = 0;
    for (int i = 0; i < nc; ++i) {
      if (ncompact > 0 && nl[i] == compact[ncompact - 1]) continue;
      compact[ncompact++] = nl[i];
    }
    if (ncompact > 1 && compact[0] == compact[ncompact - 1]) --ncompact;
    if (ncompact >= 3) add_face(out, compact, ncompact);
    for (int e = 0; e < ncompact; ++e) {
      const int a = compact[e];
      const int b = compact[(e + 1) % ncompact];
      if (is_new[a] && is_new[b] && nseg < 12) {
        seg_a[nseg] = a;
        seg_b[nseg] = b;
        ++nseg;
      }
    }
  }
  if (nseg < 3) return true;
  int used[12] = {};
  int loop[12];
  int nloop = 0;
  used[0] = 1;
  const int begin = seg_a[0];
  loop[nloop++] = begin;
  int cur = seg_b[0];
  while (cur != begin && nloop < 12) {
    loop[nloop++] = cur;
    int nxt = -1;
    for (int s = 0; s < nseg; ++s)
      if (!used[s] && seg_a[s] == cur) {
        nxt = s;
        break;
      }
    if (nxt < 0) break;
    used[nxt] = 1;
    cur = seg_b[nxt];
  }
  if (nloop < 3) return true;
  int clean[12];
  int nclean = 0;
  for (int i = 0; i < nloop; ++i) {
    if (nclean > 0 && loop[i] == clean[nclean - 1]) continue;
    clean[nclean++] = loop[i];
  }
  if (nclean > 1 && clean[0] == clean[nclean - 1]) --nclean;
  if (nclean < 3) return true;
  const Vec3 ncap =
      cross(out.v[clean[1]] - out.v[clean[0]], out.v[clean[2]] - out.v[clean[0]]);
  if (dot(ncap, normal) < 0) {
    for (int i = 0; i < nclean / 2; ++i) {
      const int swap = clean[i];
      clean[i] = clean[nclean - 1 - i];
      clean[nclean - 1 - i] = swap;
    }
  }
  out.cap = out.nf;
  add_face(out, clean, nclean);
  return true;
}

OPENPFC_HD inline int hex_tet_corner(int tet, int vertex) {
  const int id[24] = {0, 1, 3, 7, 0, 1, 7, 5, 0, 2, 7, 3,
                      0, 2, 6, 7, 0, 4, 5, 7, 0, 4, 6, 7};
  return id[tet * 4 + vertex];
}

OPENPFC_HD inline void hex_corner(Box box, int id, Vec3 &out) {
  out = {(id & 1) ? box.hi.x : box.lo.x, (id & 2) ? box.hi.y : box.lo.y,
         (id & 4) ? box.hi.z : box.lo.z};
}

} // namespace pfc::geometry
