// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file vec3.hpp
 * @brief Host/device 3-vector used by the Cartesian geometry layer.
 *
 * Physical coordinates stay in the `Domain` convention: the node `(i,j,k)`
 * sits at `origin + (i,j,k) * spacing`. This type does not introduce a
 * second grid index.
 */

#include <openpfc/kernel/data/host_device.hpp>

#include <cmath>

namespace pfc::geometry {

struct Vec3 {
  double x = 0;
  double y = 0;
  double z = 0;

  OPENPFC_HD Vec3() : x(0), y(0), z(0) {}
  OPENPFC_HD constexpr Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
};

OPENPFC_HD inline Vec3 operator+(Vec3 a, Vec3 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
OPENPFC_HD inline Vec3 operator-(Vec3 a, Vec3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
OPENPFC_HD inline Vec3 operator*(Vec3 a, double s) {
  return {a.x * s, a.y * s, a.z * s};
}
OPENPFC_HD inline Vec3 operator*(double s, Vec3 a) { return a * s; }
OPENPFC_HD inline Vec3 operator/(Vec3 a, double s) {
  return {a.x / s, a.y / s, a.z / s};
}
OPENPFC_HD inline double dot(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
OPENPFC_HD inline Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
OPENPFC_HD inline double norm2(Vec3 a) { return dot(a, a); }
OPENPFC_HD inline double norm(Vec3 a) { return sqrt(norm2(a)); }
OPENPFC_HD inline double triple(Vec3 a, Vec3 b, Vec3 c) {
  return dot(a, cross(b, c));
}

OPENPFC_HD inline double absd(double v) { return v < 0 ? -v : v; }

OPENPFC_HD inline double clamp_scalar(double v, double a, double b) {
  return v < a ? a : (v > b ? b : v);
}

OPENPFC_HD inline Vec3 clamp_box(Vec3 p, Vec3 lo, Vec3 hi) {
  return {clamp_scalar(p.x, lo.x, hi.x), clamp_scalar(p.y, lo.y, hi.y),
          clamp_scalar(p.z, lo.z, hi.z)};
}

/// Central difference. `step` is a physical length and must be positive.
template <class F>
OPENPFC_HD Vec3 numerical_gradient(const F &f, Vec3 x, double step) {
  const Vec3 dx{step, 0, 0};
  const Vec3 dy{0, step, 0};
  const Vec3 dz{0, 0, step};
  return {(f(x + dx) - f(x - dx)) / (2 * step), (f(x + dy) - f(x - dy)) / (2 * step),
          (f(x + dz) - f(x - dz)) / (2 * step)};
}

} // namespace pfc::geometry
