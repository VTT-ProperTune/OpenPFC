// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file rotation.hpp
 * @brief Proper rotation taking local material axes into physical space.
 *
 * Columns of `matrix` are the images of the local basis. A reflector
 * (`det = -1`) is rejected: orientation here is a rotation, not a
 * handedness change. The matrix is the stored representation; axis-angle
 * is only a constructor.
 */

#include <openpfc/kernel/geometry/vec3.hpp>

#include <stdexcept>

namespace pfc::geometry {

struct Rotation {
  /// Column-major. Column `j` is `matrix[j*3 + i]` (row `i`).
  double matrix[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

  OPENPFC_HD Vec3 col(int j) const {
    return {matrix[j * 3], matrix[j * 3 + 1], matrix[j * 3 + 2]};
  }

  OPENPFC_HD Vec3 apply(Vec3 local) const {
    return col(0) * local.x + col(1) * local.y + col(2) * local.z;
  }

  OPENPFC_HD Vec3 apply_transpose(Vec3 world) const {
    return {dot(col(0), world), dot(col(1), world), dot(col(2), world)};
  }
};

inline Rotation identity_rotation() { return {}; }

inline Rotation rotation_from_matrix(const double m[9]) {
  Rotation r;
  for (int i = 0; i < 9; ++i) r.matrix[i] = m[i];
  const double tol = 1e-8;
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      const double want = i == j ? 1.0 : 0.0;
      const double got = dot(r.col(i), r.col(j));
      if (std::abs(got - want) > tol) {
        throw std::invalid_argument("rotation matrix is not orthonormal");
      }
    }
  }
  if (std::abs(triple(r.col(0), r.col(1), r.col(2)) - 1.0) > tol) {
    throw std::invalid_argument("rotation matrix is a reflection or is singular");
  }
  return r;
}

/// Right-handed rotation about a nonzero axis by `radians`.
inline Rotation rotation_about(Vec3 axis, double radians) {
  const double len = norm(axis);
  if (!(len > 0) || !std::isfinite(len) || !std::isfinite(radians)) {
    throw std::invalid_argument("rotation axis must be finite and nonzero");
  }
  const Vec3 u = axis / len;
  const double c = std::cos(radians);
  const double s = std::sin(radians);
  const double v = 1.0 - c;
  // Rodrigues, columns are images of e1, e2, e3.
  const double m[9] = {
      c + u.x * u.x * v,       u.y * u.x * v + u.z * s, u.z * u.x * v - u.y * s,
      u.x * u.y * v - u.z * s, c + u.y * u.y * v,       u.z * u.y * v + u.x * s,
      u.x * u.z * v + u.y * s, u.y * u.z * v - u.x * s, c + u.z * u.z * v};
  return rotation_from_matrix(m);
}

} // namespace pfc::geometry
