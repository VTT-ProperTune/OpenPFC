// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file implicit.hpp
 * @brief Static implicit interfaces. Negative inside, positive outside,
 * zero on the interface.
 *
 * Plane (unit normal), sphere, and circle are signed-distance fields.
 * The ellipsoid and ellipse polynomials
 * `(R^T (x-c))_i^2 / a_i^2 - 1` are not: `|φ|` is not Euclidean distance.
 * Analytic normals come from `∇φ`. A vanishing gradient is reported; it
 * is not replaced by a guessed direction.
 *
 * The value is not tiled across periodic images. Call `wrap_coordinate`
 * first when a periodic copy is intended.
 */

#include <openpfc/kernel/geometry/policy.hpp>
#include <openpfc/kernel/geometry/rotation.hpp>

#include <stdexcept>

namespace pfc::geometry {

enum class Family { Plane, Sphere, Ellipsoid, Circle, Ellipse };

OPENPFC_HD inline bool is_signed_distance(Family family) {
  return family == Family::Plane || family == Family::Sphere ||
         family == Family::Circle;
}

struct Range {
  double min_phi = 0;
  double max_phi = 0;
  Vec3 at_min{};
  Vec3 at_max{};
  double grad_at_min = 0;
  double grad_at_max = 0;
};

struct Normal {
  Vec3 direction{};
  double grad_norm = 0;
  int degenerate = 1;
};

struct PackedImplicit {
  int family = 0;
  /// point, unit normal, semi-axes, then the 9 column-major rotation entries.
  double p[18]{};
};

namespace detail {

OPENPFC_HD inline void sym_mul(const double A[6], Vec3 y, double out[3]) {
  // A stored xx, yy, zz, xy, xz, yz.
  out[0] = A[0] * y.x + A[3] * y.y + A[4] * y.z;
  out[1] = A[3] * y.x + A[1] * y.y + A[5] * y.z;
  out[2] = A[4] * y.x + A[5] * y.y + A[2] * y.z;
}

OPENPFC_HD inline bool kkt_minimum(const double A[6], Vec3 center, Vec3 lo, Vec3 hi,
                                   Vec3 &at, double &phi) {
  bool found = false;
  double best = 0;
  for (int code = 0; code < 27; ++code) {
    int state[3] = {code % 3, (code / 3) % 3, code / 9};
    int fixed[3], nfixed = 0, free_id[3], nfree = 0;
    Vec3 x = center;
    bool degenerate_axis = false;
    for (int a = 0; a < 3; ++a) {
      const double span = (&hi.x)[a] - (&lo.x)[a];
      if (span <= 0) {
        (&x.x)[a] = (&lo.x)[a];
        fixed[nfixed++] = a;
        if (state[a] != 0) degenerate_axis = true;
      } else if (state[a] == 0) {
        free_id[nfree++] = a;
      } else {
        (&x.x)[a] = state[a] == 1 ? (&lo.x)[a] : (&hi.x)[a];
        fixed[nfixed++] = a;
      }
    }
    if (degenerate_axis) continue;
    double Aff[3][3]{}, rhs[3]{};
    for (int r = 0; r < nfree; ++r) {
      const int i = free_id[r];
      double row[3];
      Vec3 ei{i == 0 ? 1.0 : 0.0, i == 1 ? 1.0 : 0.0, i == 2 ? 1.0 : 0.0};
      sym_mul(A, ei, row);
      for (int c = 0; c < nfree; ++c) Aff[r][c] = row[free_id[c]];
      Vec3 yfix{};
      for (int c = 0; c < nfixed; ++c) {
        const int j = fixed[c];
        (&yfix.x)[j] = (&x.x)[j] - (&center.x)[j];
      }
      double Ay[3];
      sym_mul(A, yfix, Ay);
      rhs[r] = -Ay[i];
    }
    bool solved = true;
    double yf[3] = {};
    if (nfree == 1) {
      if (!(absd(Aff[0][0]) > 1e-18))
        solved = false;
      else
        yf[0] = rhs[0] / Aff[0][0];
    } else if (nfree == 2) {
      const double det = Aff[0][0] * Aff[1][1] - Aff[0][1] * Aff[1][0];
      if (!(absd(det) > 1e-18))
        solved = false;
      else {
        yf[0] = (rhs[0] * Aff[1][1] - Aff[0][1] * rhs[1]) / det;
        yf[1] = (Aff[0][0] * rhs[1] - Aff[1][0] * rhs[0]) / det;
      }
    } else if (nfree == 3) {
      yf[0] = yf[1] = yf[2] = 0;
    }
    if (!solved) continue;
    for (int r = 0; r < nfree; ++r) {
      const int i = free_id[r];
      (&x.x)[i] = (&center.x)[i] + yf[r];
      const double tol = 1e-9 * (1 + absd((&center.x)[i]));
      if ((&x.x)[i] < (&lo.x)[i] - tol || (&x.x)[i] > (&hi.x)[i] + tol)
        solved = false;
    }
    if (!solved) continue;
    Vec3 y = x - center;
    double Ay[3];
    sym_mul(A, y, Ay);
    bool kkt = true;
    for (int c = 0; c < nfixed; ++c) {
      const int i = fixed[c];
      if ((&hi.x)[i] - (&lo.x)[i] <= 0) continue;
      const double tol = 1e-8 * (1 + absd(Ay[i]));
      if (state[i] == 1 && Ay[i] < -tol) kkt = false;
      if (state[i] == 2 && Ay[i] > tol) kkt = false;
    }
    if (!kkt) continue;
    const double value = y.x * Ay[0] + y.y * Ay[1] + y.z * Ay[2] - 1;
    if (!found || value < best) {
      found = true;
      best = value;
      at = x;
      phi = value;
    }
  }
  return found;
}

OPENPFC_HD inline void corners_of(Vec3 lo, Vec3 hi, Vec3 out[8]) {
  for (int k = 0; k < 8; ++k) {
    out[k] = {(k & 1) ? hi.x : lo.x, (k & 2) ? hi.y : lo.y, (k & 4) ? hi.z : lo.z};
  }
}

} // namespace detail

struct Implicit {
  Family family = Family::Plane;
  Vec3 point{};
  Vec3 normal{0, 0, 1};
  Vec3 semi{1, 1, 1};
  Rotation rotation{};

  OPENPFC_HD double value(Vec3 x) const {
    if (family == Family::Plane) return dot(normal, x - point);
    if (family == Family::Sphere) return norm(x - point) - semi.x;
    if (family == Family::Circle) {
      const Vec3 d = x - point;
      return sqrt(d.x * d.x + d.y * d.y) - semi.x;
    }
    const Vec3 local = rotation.apply_transpose(x - point);
    if (family == Family::Ellipse) {
      return (local.x * local.x) / (semi.x * semi.x) +
             (local.y * local.y) / (semi.y * semi.y) - 1;
    }
    return (local.x * local.x) / (semi.x * semi.x) +
           (local.y * local.y) / (semi.y * semi.y) +
           (local.z * local.z) / (semi.z * semi.z) - 1;
  }

  OPENPFC_HD Vec3 gradient(Vec3 x) const {
    if (family == Family::Plane) return normal;
    if (family == Family::Sphere) {
      const Vec3 d = x - point;
      const double len = norm(d);
      if (!(len > 0)) return {};
      return d / len;
    }
    if (family == Family::Circle) {
      const Vec3 d = x - point;
      const double len = sqrt(d.x * d.x + d.y * d.y);
      if (!(len > 0)) return {};
      return {d.x / len, d.y / len, 0};
    }
    const Vec3 local = rotation.apply_transpose(x - point);
    Vec3 gl{2 * local.x / (semi.x * semi.x), 2 * local.y / (semi.y * semi.y), 0};
    if (family == Family::Ellipsoid) gl.z = 2 * local.z / (semi.z * semi.z);
    return rotation.apply(gl);
  }

  OPENPFC_HD Normal unit_normal(Vec3 x, const Policy &policy) const {
    const Vec3 g = gradient(x);
    const double gnorm = norm(g);
    Normal out;
    out.grad_norm = gnorm;
    if (!(gnorm > policy.gradient_absolute)) return out;
    out.direction = g / gnorm;
    out.degenerate = 0;
    return out;
  }

  OPENPFC_HD bool metric(double A[6]) const {
    if (family != Family::Ellipsoid && family != Family::Ellipse) return false;
    const double inv2[3] = {1 / (semi.x * semi.x), 1 / (semi.y * semi.y),
                            family == Family::Ellipsoid ? 1 / (semi.z * semi.z) : 0};
    for (int i = 0; i < 6; ++i) A[i] = 0;
    for (int a = 0; a < 3; ++a) {
      const Vec3 c = rotation.col(a);
      const double w = inv2[a];
      A[0] += w * c.x * c.x;
      A[1] += w * c.y * c.y;
      A[2] += w * c.z * c.z;
      A[3] += w * c.x * c.y;
      A[4] += w * c.x * c.z;
      A[5] += w * c.y * c.z;
    }
    return true;
  }

  /// Exact range of φ on the closed box. Ellipsoid/ellipse use the convex
  /// quadratic; a corner-sign test is not this query.
  OPENPFC_HD bool range_on_box(Vec3 lo, Vec3 hi, Range &out) const {
    if (family == Family::Plane || family == Family::Sphere ||
        family == Family::Circle) {
      Vec3 c[8];
      detail::corners_of(lo, hi, c);
      bool any = false;
      for (int i = 0; i < 8; ++i) {
        if (family == Family::Circle && c[i].z != lo.z && c[i].z != hi.z) continue;
        const double v = value(c[i]);
        if (!any || v < out.min_phi) {
          out.min_phi = v;
          out.at_min = c[i];
        }
        if (!any || v > out.max_phi) {
          out.max_phi = v;
          out.at_max = c[i];
        }
        any = true;
      }
      if (family == Family::Sphere || family == Family::Circle) {
        const Vec3 q = clamp_box(point, lo, hi);
        const Vec3 closest = family == Family::Circle ? Vec3{q.x, q.y, point.z} : q;
        const double vmin = value(closest);
        if (vmin < out.min_phi) {
          out.min_phi = vmin;
          out.at_min = closest;
        }
      }
      out.grad_at_min = norm(gradient(out.at_min));
      out.grad_at_max = norm(gradient(out.at_max));
      return true;
    }
    double A[6];
    if (!metric(A)) return false;
    Vec3 c[8];
    detail::corners_of(lo, hi, c);
    out.max_phi = value(c[0]);
    out.at_max = c[0];
    const int ncorner = family == Family::Ellipse ? 4 : 8;
    for (int i = 0; i < ncorner; ++i) {
      Vec3 p = c[i];
      if (family == Family::Ellipse) p.z = point.z;
      const double v = value(p);
      if (v > out.max_phi) {
        out.max_phi = v;
        out.at_max = p;
      }
    }
    Vec3 at{};
    double vmin = 0;
    Vec3 lo2 = lo;
    Vec3 hi2 = hi;
    if (family == Family::Ellipse) {
      lo2.z = point.z;
      hi2.z = point.z;
    }
    if (!detail::kkt_minimum(A, point, lo2, hi2, at, vmin)) return false;
    out.min_phi = vmin;
    out.at_min = at;
    out.grad_at_min = norm(gradient(at));
    out.grad_at_max = norm(gradient(out.at_max));
    return true;
  }

  /// Axis-aligned bounds of the body. Empty if the family is a plane.
  OPENPFC_HD bool body_bounds(Vec3 &lo, Vec3 &hi) const {
    if (family == Family::Sphere || family == Family::Circle) {
      const double r = semi.x;
      lo = {point.x - r, point.y - r, point.z - (family == Family::Sphere ? r : 0)};
      hi = {point.x + r, point.y + r, point.z + (family == Family::Sphere ? r : 0)};
      if (family == Family::Circle) {
        lo.z = point.z;
        hi.z = point.z;
      }
      return true;
    }
    if (family != Family::Ellipsoid && family != Family::Ellipse) return false;
    const double axis2[3] = {semi.x * semi.x, semi.y * semi.y,
                             family == Family::Ellipsoid ? semi.z * semi.z : 0};
    double ext[3] = {};
    for (int i = 0; i < 3; ++i) {
      const Vec3 e =
          i == 0 ? Vec3{1, 0, 0} : (i == 1 ? Vec3{0, 1, 0} : Vec3{0, 0, 1});
      const Vec3 local = rotation.apply_transpose(e);
      ext[i] = sqrt(axis2[0] * local.x * local.x + axis2[1] * local.y * local.y +
                    axis2[2] * local.z * local.z);
    }
    lo = {point.x - ext[0], point.y - ext[1], point.z - ext[2]};
    hi = {point.x + ext[0], point.y + ext[1], point.z + ext[2]};
    return true;
  }

  OPENPFC_HD double body_volume() const {
    constexpr double pi = 3.14159265358979323846;
    if (family == Family::Sphere) return 4.0 / 3.0 * pi * semi.x * semi.x * semi.x;
    if (family == Family::Circle) return pi * semi.x * semi.x;
    if (family == Family::Ellipsoid)
      return 4.0 / 3.0 * pi * semi.x * semi.y * semi.z;
    if (family == Family::Ellipse) return pi * semi.x * semi.y;
    return 0;
  }

  OPENPFC_HD double body_area() const {
    constexpr double pi = 3.14159265358979323846;
    if (family == Family::Sphere) return 4 * pi * semi.x * semi.x;
    if (family == Family::Circle) return 2 * pi * semi.x;
    return 0;
  }
};

inline Implicit make_plane(Vec3 point, Vec3 normal) {
  const double len = norm(normal);
  if (!(len > 0) || !std::isfinite(len))
    throw std::invalid_argument("plane normal must be finite and nonzero");
  Implicit s;
  s.family = Family::Plane;
  s.point = point;
  s.normal = normal / len;
  return s;
}

inline Implicit make_sphere(Vec3 center, double radius) {
  if (!(radius > 0) || !std::isfinite(radius))
    throw std::invalid_argument("sphere radius must be positive");
  Implicit s;
  s.family = Family::Sphere;
  s.point = center;
  s.semi = {radius, radius, radius};
  return s;
}

inline Implicit make_circle(Vec3 center, double radius) {
  if (!(radius > 0) || !std::isfinite(radius))
    throw std::invalid_argument("circle radius must be positive");
  Implicit s;
  s.family = Family::Circle;
  s.point = center;
  s.semi = {radius, radius, radius};
  return s;
}

inline void require_axes(Vec3 axes, int n) {
  const double a[3] = {axes.x, axes.y, axes.z};
  for (int i = 0; i < n; ++i) {
    if (!(a[i] > 0) || !std::isfinite(a[i]))
      throw std::invalid_argument("semi-axes must be positive");
  }
}

inline Implicit make_ellipsoid(Vec3 center, Vec3 semi_axes, Rotation rotation) {
  require_axes(semi_axes, 3);
  Implicit s;
  s.family = Family::Ellipsoid;
  s.point = center;
  s.semi = semi_axes;
  s.rotation = rotation;
  return s;
}

inline Implicit make_ellipse(Vec3 center, Vec3 semi_axes, Rotation rotation) {
  require_axes(semi_axes, 2);
  if (norm(rotation.col(2) - Vec3{0, 0, 1}) > 1e-8) {
    throw std::invalid_argument("ellipse rotation must preserve the z axis");
  }
  Implicit s;
  s.family = Family::Ellipse;
  s.point = center;
  s.semi = {semi_axes.x, semi_axes.y, 1};
  s.rotation = rotation;
  return s;
}

inline Implicit make_axis_aligned_ellipsoid(Vec3 center, Vec3 semi_axes) {
  return make_ellipsoid(center, semi_axes, identity_rotation());
}

OPENPFC_HD inline Side side_of(const Implicit &shape, Vec3 x, const Policy &policy,
                               double length_scale) {
  const double g = norm(shape.gradient(x));
  const double band = phi_band(policy, g, length_scale);
  const double v = shape.value(x);
  if (v < -band) return Side::Inside;
  if (v > band) return Side::Outside;
  return Side::OnInterface;
}

/// Map `x` into the periodic fundamental domain `[origin, origin+period)`.
OPENPFC_HD inline Vec3 wrap_coordinate(Vec3 x, Vec3 origin, Vec3 period,
                                       const int periodic[3]) {
  const double *xv = &x.x;
  const double *ov = &origin.x;
  const double *pv = &period.x;
  double out[3];
  for (int a = 0; a < 3; ++a) {
    out[a] = xv[a];
    if (!periodic[a] || !(pv[a] > 0)) continue;
    double t = (xv[a] - ov[a]) / pv[a];
    t = t - floor(t);
    if (t < 0) t += 1;
    out[a] = ov[a] + t * pv[a];
  }
  return {out[0], out[1], out[2]};
}

OPENPFC_HD inline PackedImplicit pack(const Implicit &shape) {
  PackedImplicit out;
  out.family = static_cast<int>(shape.family);
  out.p[0] = shape.point.x;
  out.p[1] = shape.point.y;
  out.p[2] = shape.point.z;
  out.p[3] = shape.normal.x;
  out.p[4] = shape.normal.y;
  out.p[5] = shape.normal.z;
  out.p[6] = shape.semi.x;
  out.p[7] = shape.semi.y;
  out.p[8] = shape.semi.z;
  for (int i = 0; i < 9; ++i) out.p[9 + i] = shape.rotation.matrix[i];
  return out;
}

OPENPFC_HD inline Implicit unpack(const PackedImplicit &packed) {
  Implicit shape;
  shape.family = static_cast<Family>(packed.family);
  shape.point = {packed.p[0], packed.p[1], packed.p[2]};
  shape.normal = {packed.p[3], packed.p[4], packed.p[5]};
  shape.semi = {packed.p[6], packed.p[7], packed.p[8]};
  for (int i = 0; i < 9; ++i) shape.rotation.matrix[i] = packed.p[9 + i];
  return shape;
}

} // namespace pfc::geometry
