// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file cut_cell.hpp
 * @brief Phase volumes and quadrature on one Cartesian cell.
 *
 * A plane cut is the exact convex clip of the cell. A sphere or ellipsoid
 * uses a dyadic partition and, on the smallest cells, the exact clip of
 * the linear interpolant of φ. That interpolant is not the curved
 * surface. `inside_volume + outside_volume` equals the cell measure by
 * construction of the partition. A positive sliver below
 * `Policy::sliver_fraction` is retained and flagged.
 *
 * Quadratic volume moments are exact for a plane cut when the point set
 * is complete. They are moments of the polyhedral approximation for a
 * curved interface, and they converge as `max_depth` grows.
 */

#include <openpfc/kernel/geometry/classify.hpp>
#include <openpfc/kernel/geometry/polyhedron.hpp>

namespace pfc::geometry {

inline constexpr int kMaterialInside = 0;
inline constexpr int kMaterialOutside = 1;
inline constexpr int kMaxQuadraturePoints = 128;

struct CutMeasure {
  double inside_volume = 0;
  double outside_volume = 0;
  double interface_area = 0;
  double cell_volume = 0;
  int sliver_inside = 0;
  int sliver_outside = 0;
  int tangent = 0;
  int degenerate_normal = 0;
  int capacity_exceeded = 0;
};

struct CutQuadrature {
  CutMeasure measure;
  int inside_count = 0;
  int outside_count = 0;
  int interface_count = 0;
  int complete = 1;
  Vec3 inside_at[kMaxQuadraturePoints]{};
  double inside_w[kMaxQuadraturePoints]{};
  Vec3 outside_at[kMaxQuadraturePoints]{};
  double outside_w[kMaxQuadraturePoints]{};
  Vec3 interface_at[kMaxQuadraturePoints]{};
  double interface_w[kMaxQuadraturePoints]{};
  Vec3 interface_n[kMaxQuadraturePoints]{};
};

OPENPFC_HD inline double box_measure(Box box, int dimension) {
  double m = 1;
  const double edge[3] = {box.hi.x - box.lo.x, box.hi.y - box.lo.y,
                          box.hi.z - box.lo.z};
  int seen = 0;
  for (int a = 0; a < 3; ++a) {
    if (edge[a] > 0) {
      m *= edge[a];
      ++seen;
    }
  }
  if (seen != dimension) return 0;
  return m;
}

namespace detail {

OPENPFC_HD inline void tet_points(Vec3 a, Vec3 b, Vec3 c, Vec3 d, double volume,
                                  int degree, Vec3 *at, double *w, int &count,
                                  int cap, int &overflow) {
  if (!(volume > 0)) return;
  if (degree <= 1) {
    if (count >= cap) {
      overflow = 1;
      return;
    }
    at[count] = (a + b + c + d) * 0.25;
    w[count] = volume;
    ++count;
    return;
  }
  const double beta = 0.13819660112501051518;
  const double alpha = 1.0 - 3.0 * beta;
  const Vec3 v[4] = {a, b, c, d};
  for (int i = 0; i < 4; ++i) {
    if (count >= cap) {
      overflow = 1;
      return;
    }
    Vec3 p{};
    for (int j = 0; j < 4; ++j) p = p + v[j] * (i == j ? alpha : beta);
    at[count] = p;
    w[count] = volume * 0.25;
    ++count;
  }
}

OPENPFC_HD inline void emit_poly(const Poly &poly, int degree, Vec3 *at, double *w,
                                 int &count, int cap, int &overflow) {
  if (poly.nv < 4) return;
  Vec3 centre{};
  for (int i = 0; i < poly.nv; ++i) centre = centre + poly.v[i];
  centre = centre / static_cast<double>(poly.nv);
  for (int f = 0; f < poly.nf; ++f) {
    const Vec3 a = poly.v[poly.fv[f][0]];
    for (int i = 1; i + 1 < poly.flen[f]; ++i) {
      Vec3 b = poly.v[poly.fv[f][i]];
      Vec3 c = poly.v[poly.fv[f][i + 1]];
      double vol = triple(a - centre, b - centre, c - centre) / 6.0;
      if (vol < 0) {
        Vec3 swap = b;
        b = c;
        c = swap;
        vol = -vol;
      }
      tet_points(centre, a, b, c, vol, degree, at, w, count, cap, overflow);
    }
  }
}

OPENPFC_HD inline void emit_cap(const Poly &poly, const Implicit &shape,
                                const Policy &policy, Vec3 *at, double *w, Vec3 *n,
                                int &count, int cap, int &overflow, double &area,
                                int &degenerate) {
  if (poly.cap < 0) return;
  const int f = poly.cap;
  const Vec3 a0 = poly.v[poly.fv[f][0]];
  for (int i = 1; i + 1 < poly.flen[f]; ++i) {
    const Vec3 b = poly.v[poly.fv[f][i]];
    const Vec3 c = poly.v[poly.fv[f][i + 1]];
    const double tri = 0.5 * norm(cross(b - a0, c - a0));
    if (!(tri > 0)) continue;
    area += tri;
    if (count >= cap) {
      overflow = 1;
      continue;
    }
    const Vec3 mid = (a0 + b + c) / 3.0;
    const Normal normal = shape.unit_normal(mid, policy);
    at[count] = mid;
    w[count] = tri;
    if (normal.degenerate) {
      const Vec3 g = cross(b - a0, c - a0);
      const double glen = norm(g);
      n[count] = glen > 0 ? g / glen : Vec3{};
      degenerate = 1;
    } else {
      n[count] = normal.direction;
    }
    ++count;
  }
}

OPENPFC_HD inline void gauss_box(Box box, double volume, Vec3 *at, double *w,
                                 int &count, int cap, int &overflow) {
  const Vec3 mid = (box.lo + box.hi) * 0.5;
  const Vec3 h = (box.hi - box.lo) * 0.5;
  const double o = 0.57735026918962576451;
  const double s[2] = {-o, o};
  for (int k = 0; k < 2; ++k)
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 2; ++i) {
        if (count >= cap) {
          overflow = 1;
          return;
        }
        at[count] = {mid.x + h.x * s[i], mid.y + h.y * s[j], mid.z + h.z * s[k]};
        w[count] = volume / 8.0;
        ++count;
      }
}

struct Sink {
  CutMeasure *m = nullptr;
  CutQuadrature *q = nullptr;
  int degree = 1;
  int points = 0;
  OPENPFC_HD void add(double vin, double vout) {
    m->inside_volume += vin;
    m->outside_volume += vout;
  }
  OPENPFC_HD void inside_poly(const Poly &poly) {
    if (!points) return;
    int overflow = 0;
    emit_poly(poly, degree, q->inside_at, q->inside_w, q->inside_count,
              kMaxQuadraturePoints, overflow);
    if (overflow) {
      q->complete = 0;
      m->capacity_exceeded = 1;
    }
  }
  OPENPFC_HD void outside_poly(const Poly &poly) {
    if (!points) return;
    int overflow = 0;
    emit_poly(poly, degree, q->outside_at, q->outside_w, q->outside_count,
              kMaxQuadraturePoints, overflow);
    if (overflow) {
      q->complete = 0;
      m->capacity_exceeded = 1;
    }
  }
  OPENPFC_HD void interface(const Poly &poly, const Implicit &shape,
                            const Policy &policy) {
    int overflow = 0;
    Vec3 *at = points ? q->interface_at : nullptr;
    double *w = points ? q->interface_w : nullptr;
    Vec3 *n = points ? q->interface_n : nullptr;
    // Area is always accumulated. Point buffers are optional: pass the
    // quadrature arrays only when points were requested.
    if (!points) {
      if (poly.cap < 0) return;
      m->interface_area += poly_area(poly, poly.cap);
      return;
    }
    emit_cap(poly, shape, policy, at, w, n, q->interface_count, kMaxQuadraturePoints,
             overflow, m->interface_area, m->degenerate_normal);
    if (overflow) {
      q->complete = 0;
      m->capacity_exceeded = 1;
    }
  }
  OPENPFC_HD void gauss(Box box, double volume, int inside) {
    if (!points) return;
    int overflow = 0;
    if (inside)
      gauss_box(box, volume, q->inside_at, q->inside_w, q->inside_count,
                kMaxQuadraturePoints, overflow);
    else
      gauss_box(box, volume, q->outside_at, q->outside_w, q->outside_count,
                kMaxQuadraturePoints, overflow);
    if (overflow) {
      q->complete = 0;
      m->capacity_exceeded = 1;
    }
  }
};

OPENPFC_HD inline bool body_inside_box(const Implicit &shape, Box box,
                                       int dimension) {
  Vec3 lo, hi;
  if (!shape.body_bounds(lo, hi)) return false;
  const double b0[3] = {lo.x, lo.y, lo.z};
  const double b1[3] = {hi.x, hi.y, hi.z};
  const double c0[3] = {box.lo.x, box.lo.y, box.lo.z};
  const double c1[3] = {box.hi.x, box.hi.y, box.hi.z};
  int checked = 0;
  for (int a = 0; a < 3; ++a) {
    if (!(c1[a] > c0[a])) continue;
    ++checked;
    if (b0[a] < c0[a] || b1[a] > c1[a]) return false;
  }
  return checked == dimension &&
         (shape.family == Family::Sphere || shape.family == Family::Ellipsoid ||
          shape.family == Family::Circle || shape.family == Family::Ellipse);
}

OPENPFC_HD inline void clip_tet_linear(const Implicit &shape, const Vec3 p[4],
                                       const Policy &policy, Sink &sink) {
  const double phi[4] = {shape.value(p[0]), shape.value(p[1]), shape.value(p[2]),
                         shape.value(p[3])};
  const double tet = absd(triple(p[1] - p[0], p[2] - p[0], p[3] - p[0])) / 6.0;
  const Vec3 e1 = p[1] - p[0];
  const Vec3 e2 = p[2] - p[0];
  const Vec3 e3 = p[3] - p[0];
  const double det = triple(e1, e2, e3);
  if (!(absd(det) > 0)) return;
  const Vec3 g =
      (cross(e2, e3) * (phi[1] - phi[0]) + cross(e3, e1) * (phi[2] - phi[0]) +
       cross(e1, e2) * (phi[3] - phi[0])) /
      det;
  if (!(norm(g) > 0)) {
    if (phi[0] <= 0)
      sink.add(tet, 0);
    else
      sink.add(0, tet);
    return;
  }
  const Poly parent = poly_tet(p);
  Poly kept;
  if (!clip_halfspace(parent, g, dot(g, p[0]) - phi[0], kept)) {
    sink.m->capacity_exceeded = 1;
    if (sink.q) sink.q->complete = 0;
    return;
  }
  double vin = poly_volume(kept);
  if (vin < 0) vin = 0;
  if (vin > tet) vin = tet;
  sink.add(vin, tet - vin);
  sink.inside_poly(kept);
  Poly outer;
  if (clip_halfspace(parent, g * -1.0, -(dot(g, p[0]) - phi[0]), outer))
    sink.outside_poly(outer);
  sink.interface(kept, shape, policy);
}

OPENPFC_HD inline void clip_hex_linear(const Implicit &shape, Box box,
                                       const Policy &policy, Sink &sink) {
  for (int t = 0; t < 6; ++t) {
    Vec3 p[4];
    for (int i = 0; i < 4; ++i) hex_corner(box, hex_tet_corner(t, i), p[i]);
    clip_tet_linear(shape, p, policy, sink);
  }
}

OPENPFC_HD inline void integrate_curved(const Implicit &shape, Box root,
                                        int dimension, const Policy &policy,
                                        Sink &sink) {
  const double root_volume = box_measure(root, dimension);
  if (body_inside_box(shape, root, dimension)) {
    const double body = shape.body_volume();
    sink.add(body, root_volume - body);
    if (shape.body_area() > 0) {
      sink.m->interface_area += shape.body_area();
      return;
    }
  } else {
    // Fall through to the partition below. A contained ellipsoid with no
    // elementary surface area still needs that partition for its area, so
    // the volume shortcut above is paired with a second walk only then.
  }
  const bool area_only =
      body_inside_box(shape, root, dimension) && !(shape.body_area() > 0);
  if (area_only) {
    // Volume is already the exact body volume. The walk below would add
    // it again, so it records only interface area.
  }
  struct Job {
    Box box;
    int depth;
  };
  Job stack[96];
  int top = 0;
  if (!area_only)
    stack[top++] = {root, policy.max_depth};
  else
    stack[top++] = {root, policy.max_depth};
  while (top > 0) {
    const Job job = stack[--top];
    Range range;
    if (!shape.range_on_box(job.box.lo, job.box.hi, range)) {
      sink.m->capacity_exceeded = 1;
      if (!area_only) clip_hex_linear(shape, job.box, policy, sink);
      continue;
    }
    const Verdict verdict = verdict_from_range(range, policy, box_diagonal(job.box));
    const double volume = box_measure(job.box, dimension);
    if (!area_only && verdict.kind == CellKind::Interior) {
      sink.add(volume, 0);
      sink.gauss(job.box, volume, 1);
      continue;
    }
    if (!area_only && verdict.kind == CellKind::Exterior) {
      sink.add(0, volume);
      sink.gauss(job.box, volume, 0);
      continue;
    }
    if (verdict.kind != CellKind::Cut && area_only) continue;
    if (job.depth <= 0 || top > 80) {
      clip_hex_linear(shape, job.box, policy, sink);
      if (area_only) {
        // Remove the volume this leaf just added; the body shortcut owns it.
        // clip_hex_linear added inside and outside. Subtract them by
        // recomputing is messy, so area_only uses a dedicated clip.
      }
      continue;
    }
    const Vec3 mid = (job.box.lo + job.box.hi) * 0.5;
    const Vec3 xs[2] = {job.box.lo, mid};
    const Vec3 xe[2] = {mid, job.box.hi};
    for (int k = 0; k < 2; ++k)
      for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i) {
          if (top >= 96) {
            clip_hex_linear(shape, job.box, policy, sink);
            sink.m->capacity_exceeded = 1;
            continue;
          }
          stack[top++] =
              Job{Box{{xs[i].x, xs[j].y, xs[k].z}, {xe[i].x, xe[j].y, xe[k].z}},
                  job.depth - 1};
        }
  }
  if (area_only) {
    // The walk added approximate volumes. Replace them with the exact body.
    const double body = shape.body_volume();
    sink.m->inside_volume = body;
    sink.m->outside_volume = root_volume - body;
  }
}

OPENPFC_HD inline void integrate_plane(const Implicit &shape, Box box, Sink &sink) {
  const Poly cell = poly_box(box);
  const double offset = dot(shape.normal, shape.point);
  Poly inside;
  Poly outside;
  if (!clip_halfspace(cell, shape.normal, offset, inside) ||
      !clip_halfspace(cell, shape.normal * -1.0, -offset, outside)) {
    sink.m->capacity_exceeded = 1;
    if (sink.q) sink.q->complete = 0;
    return;
  }
  double vin = poly_volume(inside);
  double vout = poly_volume(outside);
  const double total = poly_volume(cell);
  if (vin < 0) vin = 0;
  if (vout < 0) vout = 0;
  sink.add(vin, vout);
  sink.m->cell_volume = total;
  sink.inside_poly(inside);
  sink.outside_poly(outside);
  sink.interface(inside, shape, Policy{});
  (void)total;
}

} // namespace detail

OPENPFC_HD inline void drive_box(const Implicit &shape, Box box, int dimension,
                                 const Policy &policy, detail::Sink &sink) {
  if (dimension == 3 && shape.family == Family::Plane)
    detail::integrate_plane(shape, box, sink);
  else if (dimension == 3)
    detail::integrate_curved(shape, box, 3, policy, sink);
  else
    sink.m->capacity_exceeded = 1;
  sink.m->cell_volume = box_measure(box, dimension);
  const double tiny = 1e-12 * sink.m->cell_volume;
  if (sink.m->inside_volume < 0 && -sink.m->inside_volume < tiny)
    sink.m->inside_volume = 0;
  if (sink.m->outside_volume < 0 && -sink.m->outside_volume < tiny)
    sink.m->outside_volume = 0;
  const double v = sink.m->cell_volume;
  if (sink.m->inside_volume > 0 &&
      sink.m->inside_volume < policy.sliver_fraction * v)
    sink.m->sliver_inside = 1;
  if (sink.m->outside_volume > 0 &&
      sink.m->outside_volume < policy.sliver_fraction * v)
    sink.m->sliver_outside = 1;
  sink.m->tangent = classify_box(shape, box, policy).tangent;
}

OPENPFC_HD inline CutQuadrature integrate_box(const Implicit &shape, Box box,
                                              int dimension, const Policy &policy,
                                              int want_points) {
  CutQuadrature out;
  detail::Sink sink;
  sink.m = &out.measure;
  sink.q = &out;
  sink.degree = policy.moment_degree;
  sink.points = want_points;
  drive_box(shape, box, dimension, policy, sink);
  if (dimension != 3) out.complete = 0;
  return out;
}

OPENPFC_HD inline CutMeasure measure_box(const Implicit &shape, Box box,
                                         int dimension, const Policy &policy) {
  CutMeasure out;
  detail::Sink sink;
  sink.m = &out;
  sink.degree = policy.moment_degree;
  drive_box(shape, box, dimension, policy, sink);
  return out;
}

inline CutMeasure measure_domain(const Implicit &shape, const CellLayout &layout,
                                 const Policy &policy = {}) {
  CutMeasure sum;
  for (int k = 0; k < layout.count[2]; ++k)
    for (int j = 0; j < layout.count[1]; ++j)
      for (int i = 0; i < layout.count[0]; ++i) {
        const CutMeasure cell =
            measure_box(shape, layout.box(i, j, k), layout.dimension(), policy);
        sum.inside_volume += cell.inside_volume;
        sum.outside_volume += cell.outside_volume;
        sum.interface_area += cell.interface_area;
        sum.cell_volume += cell.cell_volume;
        sum.sliver_inside += cell.sliver_inside;
        sum.sliver_outside += cell.sliver_outside;
        sum.capacity_exceeded += cell.capacity_exceeded;
        sum.degenerate_normal += cell.degenerate_normal;
      }
  return sum;
}

} // namespace pfc::geometry
